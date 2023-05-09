use std::env;
use std::path::PathBuf;

fn main() {
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=c++");

        cc::Build::new()
            .files(["src/coreml/bindings.mm"].iter())
            .flag("-fobjc-arc")
            .compile("infer");
    }

    #[cfg(feature = "onnx")]
    {
        println!("cargo:rustc-link-lib=onnxruntime");

        let bindings = bindgen::Builder::default()
            .clang_arg("-Ivendor/onnxruntime/include")
            .header("src/onnx/bindings.h")
            .allowlist_var("ORT_API_VERSION")
            .allowlist_function("OrtGetApiBase")
            .allowlist_function("OrtSessionOptionsAppendExecutionProvider_CUDA")
            .allowlist_type("OrtApi")
            .layout_tests(false)
            .generate()
            .expect("unable to generate bindings");

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("unable to write bindings");
    }
}
