// Compiles the C++ TensorRT shim and links libnvinfer + cudart, but ONLY when the
// `trt` feature is enabled. Without it (the default), this is a no-op so the crate
// builds on machines that have no CUDA/TensorRT toolchain (CI, laptops).
//
// Override include/lib locations with env vars at build time:
//   CUDA_PATH       (default /usr/local/cuda)
//   TENSORRT_DIR    (default /usr   — NVIDIA's pip wheel / apt layout both work via
//                    explicit -L below; see the Dockerfile for how the wheel libs
//                    are placed on the linker path)
fn main() {
    if std::env::var_os("CARGO_FEATURE_TRT").is_none() {
        return;
    }

    let cuda = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let trt = std::env::var("TENSORRT_DIR").unwrap_or_else(|_| "/usr".to_string());

    cc::Build::new()
        .cpp(true)
        .file("cpp/trt_shim.cpp")
        .include("cpp")
        .include(format!("{cuda}/include"))
        .include(format!("{trt}/include"))
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .compile("trt_shim");

    // CUDA runtime
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-search=native={cuda}/lib");
    // TensorRT. NVIDIA's containers ship the .so under the Debian multiarch dir;
    // apt/tarball layouts use lib/lib64. Emit all so the right one resolves.
    println!("cargo:rustc-link-search=native={trt}/lib");
    println!("cargo:rustc-link-search=native={trt}/lib64");
    println!("cargo:rustc-link-search=native={trt}/lib/x86_64-linux-gnu");
    // Extra search dirs (e.g. pip-wheel tensorrt_libs), colon-separated.
    if let Ok(extra) = std::env::var("TRT_LINK_SEARCH") {
        for dir in extra.split(':').filter(|s| !s.is_empty()) {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }

    // Prefer the lean runtime if present (inference-only, no builder). Fall back to
    // the full nvinfer otherwise. The actual choice is driven by what the runtime
    // image ships; we emit the lean lib name and the Dockerfile guarantees it exists.
    let lean = std::env::var("TRT_LEAN").map(|v| v == "1").unwrap_or(false);
    if lean {
        println!("cargo:rustc-link-lib=dylib=nvinfer_lean");
    } else {
        println!("cargo:rustc-link-lib=dylib=nvinfer");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rerun-if-changed=cpp/trt_shim.cpp");
    println!("cargo:rerun-if-changed=cpp/trt_shim.h");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=TENSORRT_DIR");
    println!("cargo:rerun-if-env-changed=TRT_LEAN");
    println!("cargo:rerun-if-env-changed=TRT_LINK_SEARCH");
}
