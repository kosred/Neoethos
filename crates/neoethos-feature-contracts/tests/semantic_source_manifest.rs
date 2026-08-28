use neoethos_feature_contracts::{
    SemanticSourceEntryV1, SemanticSourceKindV1, SemanticSourceManifestV1,
};

#[test]
fn text_line_endings_are_canonical_but_semantic_bytes_are_not_normalized() {
    let lf = SemanticSourceEntryV1::from_bytes(
        "src/formula.rs",
        SemanticSourceKindV1::Utf8Text,
        b"fn formula() {\n    value\n}\n",
    )
    .expect("LF source");
    let crlf = SemanticSourceEntryV1::from_bytes(
        "src/formula.rs",
        SemanticSourceKindV1::Utf8Text,
        b"fn formula() {\r\n    value\r\n}\r\n",
    )
    .expect("CRLF source");
    let cr = SemanticSourceEntryV1::from_bytes(
        "src/formula.rs",
        SemanticSourceKindV1::Utf8Text,
        b"fn formula() {\r    value\r}\r",
    )
    .expect("CR source");
    assert_eq!(lf.payload_hash(), crlf.payload_hash());
    assert_eq!(lf.payload_hash(), cr.payload_hash());

    let whitespace_change = SemanticSourceEntryV1::from_bytes(
        "src/formula.rs",
        SemanticSourceKindV1::Utf8Text,
        b"fn formula() {\n     value\n}\n",
    )
    .expect("changed source");
    assert_ne!(lf.payload_hash(), whitespace_change.payload_hash());

    let bom = SemanticSourceEntryV1::from_bytes(
        "src/formula.rs",
        SemanticSourceKindV1::Utf8Text,
        b"\xef\xbb\xbffn formula() {\n    value\n}\n",
    )
    .expect("BOM is bytes, not silently stripped");
    assert_ne!(lf.payload_hash(), bom.payload_hash());
}

#[test]
fn manifest_order_is_portable_and_unsafe_or_colliding_paths_fail() {
    let a = SemanticSourceEntryV1::from_bytes("src/a.rs", SemanticSourceKindV1::Utf8Text, b"a\n")
        .expect("a");
    let b =
        SemanticSourceEntryV1::from_bytes("kernels/b.cu", SemanticSourceKindV1::Utf8Text, b"b\n")
            .expect("b");
    let first = SemanticSourceManifestV1::new(vec![a.clone(), b.clone()]).expect("manifest");
    let reordered = SemanticSourceManifestV1::new(vec![b, a]).expect("reordered manifest");
    assert_eq!(
        first.identity().to_hex(),
        "7e3700855860c803650d35defaa5abd9841b2948499dff35c923fdb95ae5eecd"
    );
    assert_eq!(first.identity(), reordered.identity());
    assert_eq!(first.canonical_bytes(), reordered.canonical_bytes());

    for path in [
        "../escape.rs",
        "./local.rs",
        "/absolute.rs",
        "src\\windows.rs",
        "C:/absolute.rs",
        "src//empty.rs",
    ] {
        assert!(
            SemanticSourceEntryV1::from_bytes(path, SemanticSourceKindV1::Utf8Text, b"x").is_err(),
            "unsafe path {path:?} was accepted"
        );
    }

    let upper = SemanticSourceEntryV1::from_bytes(
        "src/Formula.rs",
        SemanticSourceKindV1::Utf8Text,
        b"upper",
    )
    .expect("upper");
    let lower = SemanticSourceEntryV1::from_bytes(
        "src/formula.rs",
        SemanticSourceKindV1::Utf8Text,
        b"lower",
    )
    .expect("lower");
    assert!(SemanticSourceManifestV1::new(vec![upper, lower]).is_err());
}

#[test]
fn generated_entries_bind_generator_inputs_and_payload_kind() {
    let generated = SemanticSourceEntryV1::generated(
        "generated/capabilities.rs",
        SemanticSourceKindV1::Utf8Text,
        b"generated\r\nrow\r\n",
        "build.rs",
        vec!["kernels/a.cu".to_owned(), "kernels/b.cu".to_owned()],
    )
    .expect("generated entry");
    let changed_generator = SemanticSourceEntryV1::generated(
        "generated/capabilities.rs",
        SemanticSourceKindV1::Utf8Text,
        b"generated\nrow\n",
        "scripts/generate.rs",
        vec!["kernels/a.cu".to_owned(), "kernels/b.cu".to_owned()],
    )
    .expect("changed generator");
    assert_ne!(
        SemanticSourceManifestV1::new(vec![generated])
            .expect("manifest")
            .identity(),
        SemanticSourceManifestV1::new(vec![changed_generator])
            .expect("manifest")
            .identity()
    );
}
