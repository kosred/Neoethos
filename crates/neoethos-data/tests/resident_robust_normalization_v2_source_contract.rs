use std::{fs, path::PathBuf};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("neoethos-data must live below the workspace root")
        .to_path_buf()
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace_root().join(relative)).expect(relative)
}

fn blank_non_newline(bytes: &mut [u8], index: usize) {
    if !matches!(bytes[index], b'\n' | b'\r') {
        bytes[index] = b' ';
    }
}

/// Preserve byte offsets while removing comments and literals. Source-order
/// and balanced-body checks must inspect Rust/CUDA syntax, not tokens copied
/// into diagnostics, documentation or strings.
fn code_only(source: &str) -> String {
    let input = source.as_bytes();
    let mut output = input.to_vec();
    let mut index = 0;
    while index < input.len() {
        if input[index] == b'/' && input.get(index + 1) == Some(&b'/') {
            let mut cursor = index;
            while cursor < input.len() && input[cursor] != b'\n' {
                blank_non_newline(&mut output, cursor);
                cursor += 1;
            }
            index = cursor;
            continue;
        }
        if input[index] == b'/' && input.get(index + 1) == Some(&b'*') {
            let mut cursor = index;
            let mut depth = 0_usize;
            while cursor < input.len() {
                if input[cursor] == b'/' && input.get(cursor + 1) == Some(&b'*') {
                    depth += 1;
                    blank_non_newline(&mut output, cursor);
                    blank_non_newline(&mut output, cursor + 1);
                    cursor += 2;
                    continue;
                }
                if input[cursor] == b'*' && input.get(cursor + 1) == Some(&b'/') {
                    blank_non_newline(&mut output, cursor);
                    blank_non_newline(&mut output, cursor + 1);
                    cursor += 2;
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    continue;
                }
                blank_non_newline(&mut output, cursor);
                cursor += 1;
            }
            assert_eq!(
                depth, 0,
                "unterminated block comment in source contract input"
            );
            index = cursor;
            continue;
        }

        // Rust raw strings: r"...", r#"..."#, including the `r` inside a
        // byte raw-string prefix (`br#"..."#`).
        if input[index] == b'r' {
            let mut quote = index + 1;
            while input.get(quote) == Some(&b'#') {
                quote += 1;
            }
            if input.get(quote) == Some(&b'"') {
                let hashes = quote - index - 1;
                let mut cursor = quote + 1;
                let closing = loop {
                    let Some(relative_quote) =
                        input[cursor..].iter().position(|byte| *byte == b'"')
                    else {
                        panic!("unterminated raw string in source contract input");
                    };
                    let candidate = cursor + relative_quote;
                    let closes =
                        (0..hashes).all(|offset| input.get(candidate + 1 + offset) == Some(&b'#'));
                    if closes {
                        break candidate + 1 + hashes;
                    }
                    cursor = candidate + 1;
                };
                for cursor in index..closing {
                    blank_non_newline(&mut output, cursor);
                }
                index = closing;
                continue;
            }
        }

        if input[index] == b'"' {
            let mut cursor = index + 1;
            let mut escaped = false;
            while cursor < input.len() {
                let byte = input[cursor];
                blank_non_newline(&mut output, cursor);
                cursor += 1;
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    break;
                }
            }
            blank_non_newline(&mut output, index);
            index = cursor;
            continue;
        }

        // Do not confuse Rust lifetimes with character literals. A character
        // literal must close on this short, non-newline token.
        if input[index] == b'\'' {
            let mut cursor = index + 1;
            let mut escaped = false;
            let mut closing = None;
            while cursor < input.len() && cursor <= index + 8 && input[cursor] != b'\n' {
                let byte = input[cursor];
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'\'' {
                    closing = Some(cursor + 1);
                    break;
                } else if byte.is_ascii_whitespace() {
                    break;
                }
                cursor += 1;
            }
            if let Some(closing) = closing {
                for cursor in index..closing {
                    blank_non_newline(&mut output, cursor);
                }
                index = closing;
                continue;
            }
        }
        index += 1;
    }
    String::from_utf8(output).expect("masking source preserves UTF-8")
}

fn braced_body_after<'a>(source: &'a str, marker: &str) -> &'a str {
    let code = code_only(source);
    let marker_index = code
        .find(marker)
        .unwrap_or_else(|| panic!("missing structural marker `{marker}`"));
    let open = marker_index
        + code[marker_index..]
            .find('{')
            .unwrap_or_else(|| panic!("marker `{marker}` has no body"));
    let mut depth = 0_usize;
    for (relative, byte) in code.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    let close = open + relative;
                    return &source[open + 1..close];
                }
            }
            _ => {}
        }
    }
    panic!("marker `{marker}` has an unterminated body")
}

fn compact_code(source: &str) -> String {
    code_only(source)
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn move_only_violations(source: &str, type_name: &str) -> Vec<String> {
    let code = code_only(source);
    let declaration = format!("struct {type_name}");
    let declaration_index = code
        .find(&declaration)
        .unwrap_or_else(|| panic!("missing move-only type `{type_name}`"));
    let before = &code[..declaration_index];
    let item_attributes_start = before.rfind("\n\n").map_or(0, |index| index + 2);
    let attached_attributes = &code[item_attributes_start..declaration_index];
    let mut derived = Vec::new();
    let mut cursor = 0_usize;
    while let Some(relative_start) = attached_attributes[cursor..].find("#[derive(") {
        let derive_start = cursor + relative_start + "#[derive(".len();
        let derive_end = derive_start
            + attached_attributes[derive_start..]
                .find(")]")
                .unwrap_or_else(|| panic!("move-only type `{type_name}` has malformed derive"));
        derived.extend(
            attached_attributes[derive_start..derive_end]
                .split(',')
                .map(str::trim),
        );
        cursor = derive_end + 2;
    }
    assert!(
        !derived.is_empty(),
        "move-only type `{type_name}` has no attached derive attribute"
    );
    let compact = compact_code(source);
    let mut violations = Vec::new();
    for forbidden in ["Clone", "Copy"] {
        if derived
            .iter()
            .any(|derived| derived.rsplit("::").next() == Some(forbidden))
        {
            violations.push(format!("{type_name} derives {forbidden}"));
        }
        if compact.contains(&format!("{forbidden}for{type_name}")) {
            violations.push(format!("{type_name} implements {forbidden}"));
        }
    }
    violations
}

fn copy_to_argument(code: &str, call_index: usize) -> &str {
    let open = call_index + ".copy_to".len();
    assert_eq!(code.as_bytes()[open], b'(');
    let mut depth = 0_usize;
    for (relative, byte) in code.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return &code[open + 1..open + relative];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated copy_to call")
}

fn first_identifier(expression: &str) -> &str {
    let trimmed = expression.trim_start();
    let trimmed = trimmed
        .strip_prefix("&mut")
        .or_else(|| trimmed.strip_prefix('&'))
        .unwrap_or(trimmed)
        .trim_start();
    let end = trimmed
        .find(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .unwrap_or(trimmed.len());
    assert!(end > 0, "copy_to destination has no local binding");
    &trimmed[..end]
}

fn stack_array_bytes(function_body: &str, binding: &str) -> usize {
    let code = compact_code(function_body);
    let declaration = format!("letmut{binding}=");
    let start = code
        .find(&declaration)
        .unwrap_or_else(|| panic!("copy_to destination `{binding}` is not a local fixed array"))
        + declaration.len();
    let initializer = &code[start..];
    if initializer.starts_with("[0_u32;1]") {
        4
    } else if initializer.starts_with("[0_u64;SHA256_BYTES/std::mem::size_of::<u64>()]")
        || initializer.starts_with("[0_u8;SHA256_BYTES]")
    {
        32
    } else {
        panic!("copy_to destination `{binding}` has an unapproved extent")
    }
}

fn copy_to_sites(function_body: &str) -> Vec<(usize, String, usize)> {
    let code = compact_code(function_body);
    code.match_indices(".copy_to(")
        .map(|(index, _)| {
            let argument = copy_to_argument(&code, index);
            let binding = first_identifier(argument).to_string();
            let bytes = stack_array_bytes(function_body, &binding);
            (index, binding, bytes)
        })
        .collect()
}

fn assert_fit_digest_source(function_body: &str, site: &(usize, String, usize)) {
    assert_eq!(
        site.2, 32,
        "fit digest destination must be exactly 32 bytes"
    );
    let code = compact_code(function_body);
    let statement = code[..site.0]
        .rsplit(';')
        .next()
        .expect("fit digest copy statement");
    let receiver = first_identifier(statement);
    assert!(
        code.contains(&format!(
            "letmut{receiver}=StreamOrderedDeviceBufferV3::<u64>::uninitialized_async(plan.normalization_scratch_slots(),"
        )),
        "fit digest must come from the bounded normalization scratch allocation"
    );
    assert!(statement.starts_with(&format!("{receiver}.index(0..")));
    assert!(statement.contains(&format!("{}.len()", site.1)));
}

#[test]
fn robust_normalization_v2_freezes_the_cpu_math_and_bounded_resident_shape() {
    let cpu = source("crates/neoethos-data/src/core/normalization.rs");
    for required in [
        "training_values.sort_by(f64::total_cmp)",
        "median_sorted_f64(&training_values)",
        "median_sorted_f64(&deviations) * MAD_TO_SIGMA_F64",
        "32.0 * f64::EPSILON * max_abs.max(1.0)",
        "/ training_values.len() as f64",
        "normalized.clamp(-Z_CLIP_F64, Z_CLIP_F64)",
        "FeatureCellValidity::Degenerate",
        "FeatureCellValidity::NonFinite",
        "NORMALIZATION_TRANSFORM_SEMANTIC_VERSION: u32 = 2",
    ] {
        assert!(cpu.contains(required), "CPU authority lost `{required}`");
    }
    let oracle = source("crates/neoethos-data/tests/robust_normalization_semantic_v2_oracle.rs");
    for exact_bits in [
        "0x3ff7_b8ba_c710_cb29",
        "0x3fdb_b67a_e858_4caa",
        "0x4024_0000_0000_0000",
        "NORMALIZATION_TRANSFORM_SEMANTIC_VERSION, 2",
    ] {
        assert!(
            oracle.contains(exact_bits),
            "CPU oracle lost `{exact_bits}`"
        );
    }

    let runtime = source("crates/neoethos-gpu-cuda/src/resident_robust_normalization_v2.rs");
    for required in [
        "RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2: u32 = 2",
        "RESIDENT_ROBUST_NORMALIZATION_MAX_BATCH_COLUMNS_V2: usize = 64",
        "RESIDENT_ROBUST_NORMALIZATION_FIT_WORDS_V2: usize = 6",
        "RESIDENT_ROBUST_NORMALIZATION_FIT_BYTES_V2: usize = 48",
        "packed_validity_logical_bytes",
        "packed_validity_allocated_bytes",
        "VALIDITY_ATOMIC_ALIGNMENT_BYTES_V2",
        "fit_metadata_sha256",
        "fit_digest_d2h_bytes",
        "feature_value_d2h_bytes",
        "producer_ready_event_count",
        "producer_ready_event_synchronize_count",
        "primary_context_process_token",
        "producer_stream_process_token",
        "ready_event_process_token",
        "training_rows",
        "normalization_scratch_bytes",
        "fit_metadata_bytes",
        "training_rows != (0..canonical_training_end)",
    ] {
        assert!(
            runtime.contains(required),
            "runtime contract lost `{required}`"
        );
    }
    assert!(runtime.contains("feature_value_d2h_bytes: 0"));
    let build = source("crates/neoethos-gpu-cuda/build.rs");
    assert!(build.contains("const DEVICE_SOURCES: [&str; 13]"));
    assert!(build.contains("native/resident_robust_normalization_v2.cu"));

    let cuda = source("crates/neoethos-gpu-cuda/native/resident_robust_normalization_v2.cu");
    for required in [
        "robust_total_cmp_key_v2",
        "robust_atomic_load_word_v2",
        "robust_fill_training_v2",
        "robust_bitonic_stage_v2",
        "robust_summarize_values_v2",
        "robust_make_deviations_v2",
        "robust_finalize_fit_v2",
        "robust_apply_in_place_v2",
        "kCanonicalNanBitsV2",
        "kValidityDegenerateV2",
        "kValidityNonFiniteV2",
        "robust_fit_metadata_sha256_v2",
        "packed_validity_allocated_bytes",
        "training_end != canonical_training_end",
    ] {
        assert!(cuda.contains(required), "CUDA authority lost `{required}`");
    }
    assert!(!cuda.contains("const unsigned char packed = validity_u4[cell / 2U]"));
    assert!(!cuda.contains("unsigned int observed = *word"));
    assert!(!cuda.contains("cudaMemcpyDeviceToHost"));
}

#[test]
fn robust_normalization_is_post_pack_pre_sha_and_is_in_the_real_capability_census() {
    let store = source("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let pack = store
        .find("pub fn append_batch(")
        .expect("resident pack phase");
    let normalize = store
        .find("pub fn apply_resident_robust_normalization_v2")
        .expect("isolated resident normalization phase");
    let merkle = store
        .find("pub fn seal(")
        .expect("resident canonical SHA phase");
    assert!(pack < normalize && normalize < merkle);
    let seal = &store[merkle..];
    assert!(seal.contains("neoethos_resident_canonical_merkle_sha256_v3"));
    let normalization = &store[normalize..merkle];
    let synchronize = normalization
        .find("ready_event.synchronize()?")
        .expect("normalization event synchronization");
    let verdict = normalization
        .find(".copy_to(&mut validity_code_error)?")
        .expect("bounded 4-byte verdict");
    assert!(
        synchronize < verdict,
        "event must synchronize before verdict D2H"
    );

    let preflight =
        source("crates/neoethos-data/src/core/gpu_only_feature_workspace_preflight_v3.rs");
    assert!(preflight.contains("CURRENT_PENDING_RESIDENT_PRODUCERS_V3"));
    let pending = preflight
        .split_once("pub const CURRENT_PENDING_RESIDENT_PRODUCERS_V3:")
        .expect("pending producer census")
        .1
        .split_once("];")
        .expect("pending producer census end")
        .0;
    assert!(!pending.contains("ResidentFeatureProducerV3::RobustNormalization"));
    let data = source("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    assert!(data.contains("resident_robust_normalization_capability_v2()?"));
}

#[test]
fn data_component_receipt_binds_runtime_shape_lifetime_and_exact_run_identity() {
    let data = source("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    for required in [
        "pub(crate) struct RobustNormalizationAllocationReceiptV2",
        "pub(crate) struct RobustNormalizationLifetimeReceiptV2",
        "pub(crate) struct SealedRobustNormalizationComponentReceiptV2",
        "struct BoundRobustNormalizationComponentReceiptV2",
        "fn seal_robust_normalization_component_receipt_v2(",
        "fn bind_run_device_v2(",
        "validate_runtime_evidence",
        "runtime.fit_metadata_sha256()",
        "runtime.primary_context_process_token()",
        "runtime.producer_stream_process_token()",
        "runtime.feature_value_d2h_bytes()",
    ] {
        assert!(data.contains(required), "Data receipt lost `{required}`");
    }
    assert!(data.contains("robust_normalization.validate_working_set(&working_set)?"));
    assert!(data.contains("apply_resident_robust_normalization_v2"));
    let seal_runtime = data
        .split_once("pub(crate) fn seal_gpu_resident_feature_store_v3(")
        .expect("Data store sealing seam")
        .1;
    assert!(seal_runtime.contains("robust_normalization"));
    assert!(seal_runtime.contains(".validate_runtime_evidence(&evidence)?"));
    assert!(seal_runtime.contains("owner.sealed_steady_device_bytes()"));

    let contracts = source("crates/neoethos-gpu-contracts/src/resident_feature_store_v3.rs");
    let steady = contracts
        .split_once("let steady_device_bytes = checked_sum(")
        .expect("steady resident accounting")
        .1
        .split_once("let peak_device_bytes = checked_sum(")
        .expect("peak resident accounting")
        .0;
    assert!(
        steady.contains("request.fit_metadata_bytes"),
        "retained fit metadata must be charged to steady resident bytes"
    );

    let pending =
        source("crates/neoethos-data/src/core/gpu_only_feature_workspace_preflight_v3.rs");
    let section = pending
        .split_once("pub const CURRENT_PENDING_RESIDENT_PRODUCERS_V3:")
        .expect("pending producer census")
        .1
        .split_once("];")
        .expect("pending producer census end")
        .0;
    assert!(!section.contains("ResidentFeatureProducerV3::RobustNormalization"));
    assert!(data.contains("resident_robust_normalization_capability_v2()?"));
}

#[test]
fn robust_authority_carriers_are_structurally_move_only() {
    let split = source("crates/neoethos-data/src/core/gpu_resident_robust_normalization_v2.rs");
    let preflight =
        source("crates/neoethos-data/src/core/gpu_only_feature_workspace_preflight_v3.rs");
    let component = source("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    for (source, type_name) in [
        (&split, "SealedCanonicalRobustNormalizationSplitV2"),
        (&split, "PreparedResidentRobustNormalizationInputV2"),
        (&preflight, "PreparedGpuOnlyFeatureWorkspacePreflightV3"),
        (&component, "SealedRobustNormalizationComponentReceiptV2"),
        (&component, "BoundRobustNormalizationComponentReceiptV2"),
    ] {
        let violations = move_only_violations(source, type_name);
        assert!(
            violations.is_empty(),
            "move-only authority weakened: {}",
            violations.join(", ")
        );
    }

    // Mutation-sensitivity oracles cover separate derive attributes and both
    // manual trait implementation forms. The external RED audit additionally
    // exercises adding Clone to the production derive list.
    let derived_copy = split.replacen("#[derive(Debug)]", "#[derive(Copy)]\n#[derive(Debug)]", 1);
    assert_eq!(
        move_only_violations(&derived_copy, "SealedCanonicalRobustNormalizationSplitV2"),
        ["SealedCanonicalRobustNormalizationSplitV2 derives Copy"]
    );
    let manual_traits = format!(
        "{split}\nimpl Clone for PreparedResidentRobustNormalizationInputV2 {{\n    fn clone(&self) -> Self {{ todo!() }}\n}}\nimpl Copy for PreparedResidentRobustNormalizationInputV2 {{}}"
    );
    assert_eq!(
        move_only_violations(&manual_traits, "PreparedResidentRobustNormalizationInputV2"),
        [
            "PreparedResidentRobustNormalizationInputV2 implements Clone",
            "PreparedResidentRobustNormalizationInputV2 implements Copy",
        ]
    );
}

#[test]
fn data_invokes_exact_robust_normalization_before_assembler_seal_and_merkle() {
    let data = source("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    let entrypoint = braced_body_after(&data, "pub fn materialize_gpu_only_feature_store_v3(");
    let entrypoint = compact_code(entrypoint);
    let exact_apply = "seal_token.apply_resident_robust_normalization_v2(&mutassembler)?";
    assert_eq!(
        entrypoint.matches(exact_apply).count(),
        1,
        "Data must invoke the bound robust-normalization receipt exactly once"
    );
    let apply = entrypoint
        .find(exact_apply)
        .expect("exact Data robust-normalization call");
    let assembler_seal = entrypoint
        .find("letowner=assembler.seal()?")
        .expect("assembler seal");
    assert!(
        apply < assembler_seal,
        "Data must normalize before assembler seal enters canonical Merkle construction"
    );

    let runtime = source("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let seal = braced_body_after(
        &runtime,
        "pub fn seal(\n        mut self,\n    ) -> Result<Arc<ResidentFeatureStoreOwnerV3>",
    );
    assert_eq!(
        code_only(seal)
            .matches("neoethos_resident_canonical_merkle_sha256_v3(")
            .count(),
        1,
        "assembler seal must contain the one canonical Merkle launch"
    );
}

#[test]
fn cuda_u4_helpers_use_only_aligned_word_atomic_cas_access() {
    let cuda = source("crates/neoethos-gpu-cuda/native/resident_robust_normalization_v2.cu");
    let load = compact_code(braced_body_after(
        &cuda,
        "__device__ __forceinline__ unsigned int robust_atomic_load_word_v2(",
    ));
    assert_eq!(
        load.matches("atomicCAS(").count(),
        1,
        "u4 word reads must use exactly one atomicCAS load"
    );
    assert!(load.contains("atomicCAS(const_cast<unsignedint*>(word),0U,0U)"));
    for forbidden in ["return*word", "word[", "atomicExch(", "atomicAdd("] {
        assert!(
            !load.contains(forbidden),
            "plain/alternate word read `{forbidden}`"
        );
    }

    let read = compact_code(braced_body_after(
        &cuda,
        "__device__ __forceinline__ unsigned char robust_validity_at_v2(",
    ));
    assert!(read.contains("byte_index/sizeof(unsignedint)"));
    assert!(read.contains("reinterpret_cast<constunsignedint*>(validity_u4)+word_index"));
    assert_eq!(read.matches("robust_atomic_load_word_v2(word)").count(), 1);
    for forbidden in ["validity_u4[", "return*word", "=*word", "word["] {
        assert!(
            !read.contains(forbidden),
            "unaligned/plain u4 read `{forbidden}`"
        );
    }

    let write = compact_code(braced_body_after(
        &cuda,
        "__device__ __forceinline__ void robust_write_validity_v2(",
    ));
    assert!(write.contains("byte_index/sizeof(unsignedint)"));
    assert!(write.contains("reinterpret_cast<unsignedint*>(validity_u4)+word_index"));
    assert_eq!(write.matches("robust_atomic_load_word_v2(word)").count(), 1);
    assert_eq!(
        write
            .matches("atomicCAS(word,observed,replacement)")
            .count(),
        1,
        "u4 word writes must use exactly one compare/exchange site"
    );
    for forbidden in ["validity_u4[", "*word=replacement", "word[", "atomicExch("] {
        assert!(
            !write.contains(forbidden),
            "unaligned/plain u4 write `{forbidden}`"
        );
    }

    let entry = compact_code(braced_body_after(
        &cuda,
        "int neoethos_resident_robust_normalize_bar_major_f64_u4_v2(",
    ));
    assert!(entry.contains(
        "reinterpret_cast<std::uintptr_t>(bar_major_validity_u4)%alignof(unsignedint)!=0U"
    ));
    assert!(entry.contains("packed_validity_allocated_bytes%alignof(unsignedint)!=0U"));
    assert!(entry.contains(
        "expected_packed_validity_allocated_bytes=((packed_validity_logical_bytes+3U)/4U)*4U"
    ));
}

#[test]
fn production_d2h_is_exhaustively_whitelisted_by_structure_and_extent() {
    let runtime = source("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let code = compact_code(&runtime);
    assert_eq!(
        code.matches(".copy_to(").count(),
        4,
        "every synchronous D2H site must be structurally classified"
    );
    assert_eq!(
        code.matches("copy_to(").count(),
        4,
        "UFCS/free-function copy_to calls may not bypass method-call classification"
    );
    assert!(
        !code.contains("::copy_to"),
        "copy_to may not be aliased through UFCS"
    );
    for forbidden in [
        ".async_copy_to(",
        ".copy_to_async(",
        "cuMemcpyDtoH",
        "cuMemcpy(",
        "cudaMemcpy(",
        "cudaMemcpyDeviceToHost",
        "cudaMemcpyAsync",
    ] {
        assert!(
            !code.contains(forbidden),
            "unclassified production D2H API `{forbidden}`"
        );
    }
    for native_path in [
        "crates/neoethos-gpu-cuda/native/resident_feature_store_v3.cu",
        "crates/neoethos-gpu-cuda/native/resident_robust_normalization_v2.cu",
    ] {
        let native = compact_code(&source(native_path));
        for forbidden in [
            "cudaMemcpy(",
            "cudaMemcpyAsync(",
            "cuMemcpyDtoH",
            "cuMemcpy(",
        ] {
            assert!(
                !native.contains(forbidden),
                "native production D2H bypass `{forbidden}` in {native_path}"
            );
        }
    }

    let apply = braced_body_after(&runtime, "pub fn apply_resident_robust_normalization_v2(");
    let apply_sites = copy_to_sites(apply);
    assert_eq!(
        apply_sites.iter().map(|site| site.2).collect::<Vec<_>>(),
        [4, 32],
        "enabled normalization may read only the verdict and fit digest"
    );
    let renamed_apply = apply
        .replace("validity_code_error", "renamed_verdict")
        .replace("fit_digest_words", "renamed_digest")
        .replace("sort_scratch_bits", "renamed_scratch");
    let renamed_apply_sites = copy_to_sites(&renamed_apply);
    assert_eq!(
        renamed_apply_sites
            .iter()
            .map(|site| site.2)
            .collect::<Vec<_>>(),
        [4, 32],
        "D2H extent classification must not depend on local binding names"
    );
    assert_fit_digest_source(&renamed_apply, &renamed_apply_sites[1]);
    let apply_code = compact_code(apply);
    let first_statement = &apply_code[..apply_sites[0].0];
    assert!(
        first_statement
            .rsplit(';')
            .next()
            .is_some_and(|statement| { statement.contains("self.validity_code_error") })
    );
    assert_fit_digest_source(apply, &apply_sites[1]);

    let seal = braced_body_after(
        &runtime,
        "pub fn seal(\n        mut self,\n    ) -> Result<Arc<ResidentFeatureStoreOwnerV3>",
    );
    let seal_sites = copy_to_sites(seal);
    assert_eq!(
        seal_sites.iter().map(|site| site.2).collect::<Vec<_>>(),
        [4],
        "disabled normalization may read only the verdict during seal"
    );
    let seal_code = compact_code(seal);
    assert!(
        seal_code[..seal_sites[0].0]
            .rsplit(';')
            .next()
            .is_some_and(|statement| statement.contains("self.validity_code_error"))
    );

    let hashes = braced_body_after(&runtime, "pub fn compact_hashes_if_ready(");
    let hash_sites = copy_to_sites(hashes);
    assert_eq!(
        hash_sites.iter().map(|site| site.2).collect::<Vec<_>>(),
        [32],
        "ready store may read only the canonical Merkle root"
    );
    assert_eq!(
        copy_to_sites(&hashes.replace("canonical_content_merkle", "renamed_root"))
            .iter()
            .map(|site| site.2)
            .collect::<Vec<_>>(),
        [32],
        "Merkle D2H extent classification must not depend on its local binding name"
    );
    let hash_code = compact_code(hashes);
    assert!(
        hash_code[..hash_sites[0].0]
            .rsplit(';')
            .next()
            .is_some_and(|statement| statement.contains("self.canonical_content_merkle"))
    );
}
