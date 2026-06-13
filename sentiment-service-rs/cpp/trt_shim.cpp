// TensorRT 10.x inference-only shim. Deserializes a prebuilt engine and runs forward
// passes; it never builds engines, so the runtime image ships only libnvinfer + cudart
// (no builder, no ONNX parser). See trt_shim.h for the C ABI contract.
#include "trt_shim.h"

#include <cuda_runtime.h>
#include <NvInfer.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Static buffer for the last error message handed back across the C ABI.
thread_local std::string g_err;
const char *set_err(const std::string &msg) {
    g_err = msg;
    return g_err.c_str();
}

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        // Errors and warnings only; TensorRT is otherwise very chatty.
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

Logger g_logger;

// Minimal IEEE-754 half -> float, used only if the engine's logits output is fp16.
// The fp32-IO engines this service builds hit the fast memcpy path instead.
float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal: normalize
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13); // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

} // namespace

struct TrtEngine {
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;

    std::string in_ids_name;
    std::string in_mask_name;
    std::string out_name;
    nvinfer1::DataType in_dtype = nvinfer1::DataType::kINT64;
    nvinfer1::DataType out_dtype = nvinfer1::DataType::kFLOAT;
    int num_labels = 0;

    // Reused device buffers; grown on demand.
    void *d_ids = nullptr;
    void *d_mask = nullptr;
    void *d_out = nullptr;
    size_t cap_ids = 0;  // bytes
    size_t cap_mask = 0; // bytes
    size_t cap_out = 0;  // bytes

    // Pinned host staging buffers. Pinned (page-locked) memory makes cudaMemcpyAsync
    // a true async DMA instead of the synchronous bounce-through-staging path that
    // pageable memory forces, and lets us reuse one allocation across calls instead
    // of heap-allocating a std::vector every forward.
    void *h_ids = nullptr;   // sized to in_bytes
    void *h_mask = nullptr;  // sized to in_bytes
    void *h_out = nullptr;   // sized to out_bytes (raw engine output dtype)
    size_t cap_h_ids = 0;
    size_t cap_h_mask = 0;
    size_t cap_h_out = 0;

    // Cache the last-bound shape + device addresses so we only call the (comparatively
    // expensive) context setters when something actually changes. After warmup the
    // device buffers stop growing, so setTensorAddress is then never called again and
    // setInputShape only when the chunk's [batch, seq] differs.
    int last_batch = -1;
    int last_seq = -1;
    void *bound_ids = nullptr;
    void *bound_mask = nullptr;
    void *bound_out = nullptr;

    ~TrtEngine() {
        if (d_ids) cudaFree(d_ids);
        if (d_mask) cudaFree(d_mask);
        if (d_out) cudaFree(d_out);
        if (h_ids) cudaFreeHost(h_ids);
        if (h_mask) cudaFreeHost(h_mask);
        if (h_out) cudaFreeHost(h_out);
        if (stream) cudaStreamDestroy(stream);
        delete context;
        delete engine;
        delete runtime;
    }
};

static bool ensure_capacity(void **buf, size_t *cap, size_t needed, std::string *err) {
    if (needed <= *cap) return true;
    if (*buf) cudaFree(*buf);
    *buf = nullptr;
    cudaError_t st = cudaMalloc(buf, needed);
    if (st != cudaSuccess) {
        *err = std::string("cudaMalloc failed: ") + cudaGetErrorString(st);
        *cap = 0;
        return false;
    }
    *cap = needed;
    return true;
}

static bool ensure_pinned(void **buf, size_t *cap, size_t needed, std::string *err) {
    if (needed <= *cap) return true;
    if (*buf) cudaFreeHost(*buf);
    *buf = nullptr;
    cudaError_t st = cudaHostAlloc(buf, needed, cudaHostAllocDefault);
    if (st != cudaSuccess) {
        *err = std::string("cudaHostAlloc failed: ") + cudaGetErrorString(st);
        *cap = 0;
        return false;
    }
    *cap = needed;
    return true;
}

static size_t dtype_size(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        default: return 4;
    }
}

extern "C" TrtEngine *trt_engine_load(const char *engine_path, const char **err) {
    auto fail = [&](const std::string &m) -> TrtEngine * {
        if (err) *err = set_err(m);
        return nullptr;
    };

    FILE *f = std::fopen(engine_path, "rb");
    if (!f) return fail(std::string("cannot open engine: ") + engine_path);
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        std::fclose(f);
        return fail("engine file is empty");
    }
    std::vector<char> blob(size);
    size_t read = std::fread(blob.data(), 1, size, f);
    std::fclose(f);
    if ((long)read != size) return fail("short read on engine file");

    TrtEngine *h = new TrtEngine();
    h->runtime = nvinfer1::createInferRuntime(g_logger);
    if (!h->runtime) { delete h; return fail("createInferRuntime failed"); }

    h->engine = h->runtime->deserializeCudaEngine(blob.data(), blob.size());
    if (!h->engine) { delete h; return fail("deserializeCudaEngine failed"); }

    h->context = h->engine->createExecutionContext();
    if (!h->context) { delete h; return fail("createExecutionContext failed"); }

    cudaError_t st = cudaStreamCreate(&h->stream);
    if (st != cudaSuccess) {
        delete h;
        return fail(std::string("cudaStreamCreate failed: ") + cudaGetErrorString(st));
    }

    // Discover IO tensors by name + mode. Inputs are matched by name; the single
    // output is taken as the logits tensor.
    int n = h->engine->getNbIOTensors();
    for (int i = 0; i < n; i++) {
        const char *name = h->engine->getIOTensorName(i);
        auto mode = h->engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            std::string nm(name);
            if (nm == "attention_mask") {
                h->in_mask_name = nm;
            } else if (nm == "input_ids") {
                h->in_ids_name = nm;
            } else if (h->in_ids_name.empty()) {
                h->in_ids_name = nm; // fallback: first input is ids
            } else if (h->in_mask_name.empty()) {
                h->in_mask_name = nm; // fallback: second input is mask
            }
            h->in_dtype = h->engine->getTensorDataType(name);
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (h->out_name.empty() || std::string(name) == "logits") {
                h->out_name = name;
                h->out_dtype = h->engine->getTensorDataType(name);
            }
        }
    }
    if (h->in_ids_name.empty() || h->in_mask_name.empty() || h->out_name.empty()) {
        delete h;
        return fail("engine missing expected IO tensors (input_ids/attention_mask/logits)");
    }

    // num_labels = last dim of the output shape (batch dim is dynamic / -1).
    nvinfer1::Dims od = h->engine->getTensorShape(h->out_name.c_str());
    if (od.nbDims >= 1) h->num_labels = (int)od.d[od.nbDims - 1];

    if (err) *err = nullptr;
    return h;
}

extern "C" int trt_engine_num_labels(const TrtEngine *engine) {
    return engine ? engine->num_labels : 0;
}

extern "C" int trt_engine_infer(TrtEngine *h,
                                const int32_t *input_ids,
                                const int32_t *attention_mask,
                                int batch,
                                int seq,
                                float *out_logits,
                                int num_labels,
                                const char **err) {
    auto fail = [&](const std::string &m) -> int {
        if (err) *err = set_err(m);
        return 1;
    };
    if (!h) return fail("null engine handle");

    const size_t count = (size_t)batch * (size_t)seq;
    const size_t in_elt = dtype_size(h->in_dtype);
    const size_t in_bytes = count * in_elt;
    const size_t out_elt = dtype_size(h->out_dtype);
    const size_t out_count = (size_t)batch * (size_t)num_labels;
    const size_t out_bytes = out_count * out_elt;

    // Device slabs (input_ids/attention_mask share the [batch, seq] shape) + the
    // pinned host staging buffers, all grown on demand and reused across calls.
    std::string e;
    if (!ensure_capacity(&h->d_ids, &h->cap_ids, in_bytes, &e)) return fail(e);
    if (!ensure_capacity(&h->d_mask, &h->cap_mask, in_bytes, &e)) return fail(e);
    if (!ensure_capacity(&h->d_out, &h->cap_out, out_bytes, &e)) return fail(e);
    if (!ensure_pinned(&h->h_ids, &h->cap_h_ids, in_bytes, &e)) return fail(e);
    if (!ensure_pinned(&h->h_mask, &h->cap_h_mask, in_bytes, &e)) return fail(e);
    if (!ensure_pinned(&h->h_out, &h->cap_h_out, out_bytes, &e)) return fail(e);

    // Stage inputs into pinned memory, converting int32 -> engine input dtype in place.
    if (h->in_dtype == nvinfer1::DataType::kINT64) {
        int64_t *ids = (int64_t *)h->h_ids;
        int64_t *mask = (int64_t *)h->h_mask;
        for (size_t i = 0; i < count; i++) {
            ids[i] = (int64_t)input_ids[i];
            mask[i] = (int64_t)attention_mask[i];
        }
    } else { // engine takes int32 directly
        std::memcpy(h->h_ids, input_ids, in_bytes);
        std::memcpy(h->h_mask, attention_mask, in_bytes);
    }

    cudaError_t st;
    st = cudaMemcpyAsync(h->d_ids, h->h_ids, in_bytes, cudaMemcpyHostToDevice, h->stream);
    if (st != cudaSuccess) return fail(std::string("H2D ids: ") + cudaGetErrorString(st));
    st = cudaMemcpyAsync(h->d_mask, h->h_mask, in_bytes, cudaMemcpyHostToDevice, h->stream);
    if (st != cudaSuccess) return fail(std::string("H2D mask: ") + cudaGetErrorString(st));

    // Only re-set the shape when this chunk's [batch, seq] differs from the last call.
    if (batch != h->last_batch || seq != h->last_seq) {
        nvinfer1::Dims2 shape(batch, seq);
        if (!h->context->setInputShape(h->in_ids_name.c_str(), shape))
            return fail("setInputShape(input_ids) failed");
        if (!h->context->setInputShape(h->in_mask_name.c_str(), shape))
            return fail("setInputShape(attention_mask) failed");
        h->last_batch = batch;
        h->last_seq = seq;
    }

    // Only rebind a tensor address when its device buffer was (re)allocated. After the
    // buffers reach their high-water mark this never fires again.
    if (h->bound_ids != h->d_ids) {
        if (!h->context->setTensorAddress(h->in_ids_name.c_str(), h->d_ids))
            return fail("setTensorAddress(input_ids) failed");
        h->bound_ids = h->d_ids;
    }
    if (h->bound_mask != h->d_mask) {
        if (!h->context->setTensorAddress(h->in_mask_name.c_str(), h->d_mask))
            return fail("setTensorAddress(attention_mask) failed");
        h->bound_mask = h->d_mask;
    }
    if (h->bound_out != h->d_out) {
        if (!h->context->setTensorAddress(h->out_name.c_str(), h->d_out))
            return fail("setTensorAddress(logits) failed");
        h->bound_out = h->d_out;
    }

    if (!h->context->enqueueV3(h->stream))
        return fail("enqueueV3 failed");

    // D2H into pinned memory, sync, then convert to fp32 for the caller.
    st = cudaMemcpyAsync(h->h_out, h->d_out, out_bytes, cudaMemcpyDeviceToHost, h->stream);
    if (st != cudaSuccess) return fail(std::string("D2H logits: ") + cudaGetErrorString(st));
    st = cudaStreamSynchronize(h->stream);
    if (st != cudaSuccess) return fail(std::string("stream sync: ") + cudaGetErrorString(st));

    if (h->out_dtype == nvinfer1::DataType::kFLOAT) {
        std::memcpy(out_logits, h->h_out, out_bytes);
    } else if (h->out_dtype == nvinfer1::DataType::kHALF) {
        const uint16_t *half = (const uint16_t *)h->h_out;
        for (size_t i = 0; i < out_count; i++) out_logits[i] = half_to_float(half[i]);
    } else {
        return fail("unsupported logits output dtype");
    }

    if (err) *err = nullptr;
    return 0;
}

extern "C" void trt_engine_free(TrtEngine *engine) {
    delete engine;
}
