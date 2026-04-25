use std::path::PathBuf;

fn main() {
    let protoc_path = protoc_bin_vendored::protoc_bin_path().unwrap();
    let protoc_dir = protoc_path.parent().unwrap();

    // Add protoc directory to PATH because protobuf-codegen v4.31 hardcodes "protoc" command
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = std::env::split_paths(&path).collect::<Vec<_>>();
    paths.insert(0, protoc_dir.to_path_buf());
    let new_path = std::env::join_paths(paths).unwrap();
    unsafe {
        std::env::set_var("PATH", new_path);
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let proto_temp_dir = out_dir.join("proto_temp");
    std::fs::create_dir_all(&proto_temp_dir).unwrap();

    // Copy .proto files to temp dir
    let files = [
        "OpenApiCommonModelMessages.proto",
        "OpenApiModelMessages.proto",
        "OpenApiCommonMessages.proto",
        "OpenApiMessages.proto",
    ];
    for file in &files {
        std::fs::copy(format!("proto/{}", file), proto_temp_dir.join(file)).unwrap();
    }

    let gen_dir = out_dir.join("protobuf_generated");
    std::fs::create_dir_all(&gen_dir).unwrap();

    protobuf_codegen::CodeGen::new()
        .include(&proto_temp_dir)
        .input("OpenApiCommonModelMessages.proto")
        .input("OpenApiModelMessages.proto")
        .input("OpenApiCommonMessages.proto")
        .input("OpenApiMessages.proto")
        .output_dir(&gen_dir)
        .generate_and_compile()
        .expect("protobuf codegen failed");

    // Post-process generated files for Rust 2024 compatibility (unsafe extern blocks)
    fn patch_dir(dir: &std::path::Path) {
        if !dir.exists() {
            return;
        }
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                patch_dir(&path);
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                let content = std::fs::read_to_string(&path).unwrap();
                let patched = content.replace("extern \"C\" {", "unsafe extern \"C\" {");
                std::fs::write(&path, patched).unwrap();
            }
        }
    }
    patch_dir(&gen_dir);
}
