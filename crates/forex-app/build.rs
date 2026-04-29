use std::io::Write;
use std::path::{Path, PathBuf};

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

    generate_protobuf_sources(&proto_temp_dir, &gen_dir, &files);
    compile_generated_protobuf_sources(&gen_dir, &files);

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
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let content = std::fs::read_to_string(&path).unwrap();
                let patched = content.replace("extern \"C\" {", "unsafe extern \"C\" {");
                std::fs::write(&path, patched).unwrap();
            }
        }
    }
    patch_dir(&gen_dir);
}

fn generate_protobuf_sources(proto_temp_dir: &Path, gen_dir: &Path, files: &[&str]) {
    let crate_mapping_path = gen_dir.join("crate_mapping.txt");
    std::fs::File::create(&crate_mapping_path)
        .and_then(|mut file| file.write_all(b""))
        .expect("failed to create protobuf crate mapping");

    let mut cmd = std::process::Command::new("protoc");
    for file in files {
        cmd.arg(file);
    }
    cmd.arg(format!("--rust_out={}", gen_dir.display()))
        .arg("--rust_opt=experimental-codegen=enabled,kernel=upb")
        .arg(format!("--upb_minitable_out={}", gen_dir.display()))
        .arg(format!("--proto_path={}", proto_temp_dir.display()))
        .arg(format!(
            "--rust_opt=crate_mapping={}",
            crate_mapping_path.display()
        ));

    println!("cargo:rerun-if-changed={}", proto_temp_dir.display());
    let output = cmd.output().expect("failed to run protoc");
    if !output.status.success() {
        panic!(
            "protobuf codegen failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn compile_generated_protobuf_sources(gen_dir: &Path, files: &[&str]) {
    let mut cc_build = cc::Build::new();
    cc_build
        .include(
            std::env::var_os("DEP_UPB_INCLUDE")
                .expect("DEP_UPB_INCLUDE should be set by the protobuf crate"),
        )
        .include(gen_dir);

    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        cc_build.flag("-std=c99");
    }

    for file in files {
        let c_file = generated_c_file(gen_dir, file);
        if !c_file.exists() {
            panic!(
                "expected generated file {} does not exist",
                c_file.display()
            );
        }
        println!("cargo:rerun-if-changed={}", c_file.display());
        cc_build.file(c_file);
    }

    cc_build.compile(&format!(
        "{}_upb_gen_code",
        std::env::var("CARGO_PKG_NAME").unwrap()
    ));
}

fn generated_c_file(gen_dir: &Path, proto_file: &str) -> PathBuf {
    let mut path = PathBuf::from(proto_file);
    assert!(path.set_extension("upb_minitable.c"));
    gen_dir.join(path)
}
