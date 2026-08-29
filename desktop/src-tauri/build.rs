fn main() {
    println!("cargo:rerun-if-env-changed=NEOETHOS_TAURI_RELEASE_BUNDLE");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
        && std::env::var("NEOETHOS_TAURI_RELEASE_BUNDLE").as_deref() == Ok("1")
    {
        // Tauri installs generic Linux bundle resources under
        // /usr/lib/<productName>, while the executable lives in /usr/bin.
        // Bind the packaged desktop executable to that exact relative runtime
        // directory so CatBoost/XGBoost resolve without LD_LIBRARY_PATH or a
        // build-tree path. `$ORIGIN` is evaluated by the ELF loader. The
        // release-only environment gate prevents an ordinary development
        // binary from silently loading a stale system-installed runtime.
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib/NeoEthos");
    }
    tauri_build::build()
}
