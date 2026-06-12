// C ABI over TensorRT + CUDA so the Rust side never touches the C++ API directly.
// All CUDA memory, streams, and the (non-thread-safe) execution context live behind
// this boundary; the caller (Rust) serializes access with a Mutex.
#ifndef TRT_SHIM_H
#define TRT_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TrtEngine TrtEngine;

// Load + deserialize a serialized engine (.plan) from disk and create one execution
// context. Returns NULL on failure; on failure *err (if non-NULL) points to a
// static, NUL-terminated message (do not free).
TrtEngine *trt_engine_load(const char *engine_path, const char **err);

// Number of output classes (logits.shape[-1]) the engine produces. 0 if unknown.
int trt_engine_num_labels(const TrtEngine *engine);

// Run one forward pass. input_ids / attention_mask are row-major [batch, seq] int32
// (the shim converts to the engine's declared input dtype — int32 or int64 — on the
// host before the H2D copy). out_logits is caller-allocated, length batch*num_labels,
// and receives fp32 logits (the shim converts from fp16 if the engine output is half).
// Returns 0 on success, non-zero on failure (message via *err, static, do not free).
int trt_engine_infer(TrtEngine *engine,
                     const int32_t *input_ids,
                     const int32_t *attention_mask,
                     int batch,
                     int seq,
                     float *out_logits,
                     int num_labels,
                     const char **err);

void trt_engine_free(TrtEngine *engine);

#ifdef __cplusplus
}
#endif

#endif // TRT_SHIM_H
