//! Native TensorRT backend via the C++ shim (cpp/trt_shim.cpp). Compiled only with
//! the `trt` feature. The TensorRT execution context is NOT thread-safe and the shim
//! holds shared device buffers, so every forward pass is serialized with a Mutex —
//! the same correctness constraint the Python service enforces (see app/model.py's
//! TensorRTRunner lock). Throughput-overlap (streams/dynamic batching) is a deliberate
//! follow-up, not part of this first port.

use std::ffi::{c_char, c_int, CStr, CString};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Result};

use super::Backend;

#[repr(C)]
struct TrtEngineHandle {
    _private: [u8; 0],
}

extern "C" {
    fn trt_engine_load(engine_path: *const c_char, err: *mut *const c_char) -> *mut TrtEngineHandle;
    fn trt_engine_num_labels(engine: *const TrtEngineHandle) -> c_int;
    fn trt_engine_infer(
        engine: *mut TrtEngineHandle,
        input_ids: *const i32,
        attention_mask: *const i32,
        batch: c_int,
        seq: c_int,
        out_logits: *mut f32,
        num_labels: c_int,
        err: *mut *const c_char,
    ) -> c_int;
    fn trt_engine_free(engine: *mut TrtEngineHandle);
}

/// Wraps the opaque C handle. The pointer is only ever dereferenced by the shim while
/// we hold `lock`, so the combination is Send + Sync.
struct Handle(*mut TrtEngineHandle);
unsafe impl Send for Handle {}

pub struct TrtBackend {
    handle: Mutex<Handle>,
    num_labels: usize,
}

unsafe fn err_string(err: *const c_char, fallback: &str) -> String {
    if err.is_null() {
        fallback.to_string()
    } else {
        CStr::from_ptr(err).to_string_lossy().into_owned()
    }
}

impl TrtBackend {
    pub fn load(engine_path: &str) -> Result<Self> {
        let c_path = CString::new(engine_path)
            .map_err(|_| anyhow!("engine path contains a NUL byte: {engine_path}"))?;
        let mut err: *const c_char = std::ptr::null();
        let ptr = unsafe { trt_engine_load(c_path.as_ptr(), &mut err) };
        if ptr.is_null() {
            let msg = unsafe { err_string(err, "trt_engine_load returned null") };
            bail!("failed to load TensorRT engine {engine_path}: {msg}");
        }
        let n = unsafe { trt_engine_num_labels(ptr) };
        let num_labels = if n > 0 { n as usize } else { 3 };
        tracing::info!(engine = %engine_path, num_labels, "loaded TensorRT engine");
        Ok(Self {
            handle: Mutex::new(Handle(ptr)),
            num_labels,
        })
    }
}

impl Backend for TrtBackend {
    fn infer(
        &self,
        input_ids: &[i32],
        attention_mask: &[i32],
        batch: usize,
        seq: usize,
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(input_ids.len(), batch * seq);
        debug_assert_eq!(attention_mask.len(), batch * seq);

        let mut out = vec![0.0_f32; batch * self.num_labels];
        let guard = self
            .handle
            .lock()
            .map_err(|_| anyhow!("TensorRT backend mutex poisoned"))?;
        let mut err: *const c_char = std::ptr::null();
        let rc = unsafe {
            trt_engine_infer(
                guard.0,
                input_ids.as_ptr(),
                attention_mask.as_ptr(),
                batch as c_int,
                seq as c_int,
                out.as_mut_ptr(),
                self.num_labels as c_int,
                &mut err,
            )
        };
        if rc != 0 {
            let msg = unsafe { err_string(err, "trt_engine_infer failed") };
            bail!("TensorRT inference failed: {msg}");
        }
        Ok(out)
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn name(&self) -> &'static str {
        "cuda-tensorrt"
    }
}

impl Drop for TrtBackend {
    fn drop(&mut self) {
        if let Ok(guard) = self.handle.lock() {
            unsafe { trt_engine_free(guard.0) };
        }
    }
}
