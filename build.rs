use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-lib=onnxruntime");

    let bindings = bindgen::Builder::default()
        .header("src/bindings.h")
        .whitelist_var("ORT_API_VERSION")
        .whitelist_function("OrtGetApiBase")
        .whitelist_type("OrtApi")
        .layout_tests(false)
        .generate()
        .expect("unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("unable to write bindings");
}
