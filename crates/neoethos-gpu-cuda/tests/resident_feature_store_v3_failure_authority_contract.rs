use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn crate_root() -> PathBuf {
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest_dir);
    }

    let current = env::current_dir().expect("current directory must be available");
    if current.join("src/resident_feature_store_v3.rs").is_file() {
        current
    } else {
        current.join("crates/neoethos-gpu-cuda")
    }
}

fn read(relative: &str) -> String {
    let path = crate_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn runtime_source() -> String {
    read("src/resident_feature_store_v3.rs")
}

fn contracts_source() -> String {
    let path = crate_root().join("../neoethos-gpu-contracts/src/resident_feature_store_v3.rs");
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn native_source() -> String {
    let root = crate_root();
    for relative in [
        "native/resident_feature_store_v3.cu",
        "native/resident_feature_store_v1.cu",
    ] {
        let path = root.join(relative);
        if path.is_file() {
            return fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        }
    }
    panic!("missing resident feature-store native V3 implementation");
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing source boundary {start:?}"));
    let tail = &source[start_index..];
    let end_index = tail
        .find(end)
        .unwrap_or_else(|| panic!("missing source boundary {end:?} after {start:?}"));
    &tail[..end_index]
}

fn item_block<'a>(source: &'a str, marker: &str) -> &'a str {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing source item {marker:?}"));
    let tail = &source[start..];
    let open = tail
        .find('{')
        .unwrap_or_else(|| panic!("source item {marker:?} has no body"));
    let mut depth = 0_usize;
    for (offset, byte) in tail[open..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &tail[..open + offset + 1];
                }
            }
            _ => {}
        }
    }
    panic!("source item {marker:?} has an unterminated body");
}

fn position(source: &str, needle: &str) -> usize {
    source
        .find(needle)
        .unwrap_or_else(|| panic!("missing required authority token {needle:?}"))
}

fn first_position(source: &str, needles: &[&str]) -> usize {
    needles
        .iter()
        .filter_map(|needle| source.find(needle))
        .min()
        .unwrap_or_else(|| panic!("missing every accepted authority token {needles:?}"))
}

fn require_all(source: &str, needles: &[&str]) {
    for needle in needles {
        assert!(
            source.contains(needle),
            "missing required authority token {needle:?}"
        );
    }
}

fn rust_sources_below(root: &Path) -> Vec<(PathBuf, String)> {
    let mut pending = vec![root.to_path_buf()];
    let mut sources = Vec::new();
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        {
            let entry = entry.expect("source directory entry must be readable");
            let entry_path = entry.path();
            if entry_path.is_dir() {
                pending.push(entry_path);
            } else if entry_path
                .extension()
                .is_some_and(|extension| extension == "rs")
            {
                let source = fs::read_to_string(&entry_path).unwrap_or_else(|error| {
                    panic!("failed to read {}: {error}", entry_path.display())
                });
                sources.push((entry_path, source));
            }
        }
    }
    sources
}

#[test]
fn append_failure_guard_owns_batch_before_any_refusal_or_fallible_cuda_step() {
    let source = runtime_source();
    let append = section(
        &source,
        "pub fn append_batch(",
        "pub fn try_retire_completed_batch",
    );

    let guard = first_position(
        append,
        &[
            "ResidentBatchFailureGuardV3::new",
            "ResidentAppendFailureGuardV3::new",
            "ResidentAppendTransactionV3::new",
        ],
    );
    let first_refusal = position(append, "if self.pending_batch.is_some()");
    let context_selection = position(append, "CurrentContext::set_current");
    let native_pack = position(
        append,
        "neoethos_resident_pack_batch_to_bar_major_f64_u4_v3(",
    );
    let event_record = first_position(
        append,
        &[
            "batch_ready_event.record",
            "transaction.ready_event().record",
        ],
    );
    let disarm = first_position(
        append,
        &[
            "disarm_after_pack_event",
            "failure_guard.disarm()",
            "transaction.disarm()",
        ],
    );

    assert!(
        guard < first_refusal && guard < context_selection,
        "the batch owner guard must be armed before every refusal, `?`, CUDA call, or allocation"
    );
    assert!(
        native_pack < event_record && event_record < disarm,
        "the guard may transfer ownership only after the real pack event was recorded"
    );
    assert!(
        append[guard..first_refusal].contains("batch"),
        "the early guard must own the producer batch, not only later pointer metadata"
    );
}

#[test]
fn consumer_completion_failure_guard_retains_live_owner_until_wait_is_enqueued() {
    let source = runtime_source();
    let record = section(
        &source,
        "pub fn record_consumer_completion(",
        "impl Drop for ResidentFeatureStoreImportV3",
    );

    let set_context = position(record, "CurrentContext::set_current");
    let create_event = position(record, "OwnedCudaEventV3::new()");
    let record_event = position(record, "consumer_completion_event.record");
    let queue_wait = position(record, "consumer_completion_event.enqueue_wait");
    assert!(set_context < create_event && create_event < record_event && record_event < queue_wait);

    if record.contains("ResidentConsumerCompletionFailureGuardV3::new") {
        let guard = position(record, "ResidentConsumerCompletionFailureGuardV3::new");
        let take_owner = position(record, "self.owner.take()");
        let disarm = position(record, "failure_guard.disarm_after_wait");
        assert!(guard < take_owner && guard < set_context && queue_wait < disarm);
        let guard_drop = item_block(
            &source,
            "impl Drop for ResidentConsumerCompletionFailureGuardV3",
        );
        require_all(
            guard_drop,
            &[
                "std::mem::forget",
                "owner",
                "consumer_context",
                "consumer_stream",
            ],
        );
    } else {
        let compact_record = record
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        let compact_queue_wait =
            position(&compact_record, "consumer_completion_event.enqueue_wait");
        for take in [
            "self.owner.take()",
            "self.consumer_context.take()",
            "self.consumer_stream.take()",
        ] {
            assert!(
                compact_queue_wait < position(&compact_record, take),
                "Import must retain {take:?} until the consumer event and producer wait succeed"
            );
        }
        let import_drop = item_block(&source, "impl Drop for ResidentFeatureStoreImportV3");
        require_all(
            import_drop,
            &[
                "std::mem::forget(owner)",
                "std::mem::forget(context)",
                "std::mem::forget(stream)",
            ],
        );
    }
}

#[test]
fn runtime_recomputes_exact_metadata_and_same_context_free_memory_before_allocation() {
    let source = runtime_source();
    require_all(
        &source,
        &[
            "mem_get_info()",
            "runtime_pointer_and_schema_metadata_bytes_v3",
            "pointer_and_schema_metadata_bytes()",
            "allocator_context_reserve_bytes()",
            "reserve_policy_id()",
            "remaining_peak_after_parent_bytes()",
            "RuntimeFreeMemoryChanged",
            "max_live_runtime_metadata_bytes",
            "pre_materialization_free_bytes_snapshot",
            "post_parent_free_bytes_snapshot",
        ],
    );

    let constructor = section(
        &source,
        "impl ResidentFeatureStoreAssemblerV3",
        "pub fn append_batch(",
    );
    let set_context = position(constructor, "CurrentContext::set_current");
    let query_free = position(constructor, "mem_get_info()");
    let first_allocation = first_position(
        constructor,
        &["uninitialized_async(", "zeroed_async(", "from_slice_async("],
    );
    assert!(
        set_context < query_free && query_free < first_allocation,
        "post-parent free memory must be queried in the exact selected context before the first final-store allocation"
    );
    require_all(
        constructor,
        &[
            "remaining_peak_after_parent_bytes > observed_available",
            "working_set.remaining_peak_after_parent_bytes()",
        ],
    );
    for forbidden_equality in [
        "observed_free_bytes == working_set.device_free_bytes_snapshot()",
        "observed_free_bytes != working_set.device_free_bytes_snapshot()",
        "working_set.device_free_bytes_snapshot() == observed_free_bytes",
        "working_set.device_free_bytes_snapshot() != observed_free_bytes",
    ] {
        assert!(
            !constructor.contains(forbidden_equality),
            "phase-one free memory cannot equal the later post-parent snapshot via {forbidden_equality:?}"
        );
    }

    let contracts = contracts_source();
    let remaining_peak = item_block(
        &contracts,
        "pub const fn remaining_peak_after_parent_bytes(&self)",
    );
    require_all(
        remaining_peak,
        &["self.peak_device_bytes - self.parent_dataset_bytes"],
    );

    let append = section(
        &source,
        "pub fn append_batch(",
        "pub fn try_retire_completed_batch",
    );
    require_all(
        append,
        &[
            "runtime_pointer_and_schema_metadata_bytes_v3",
            "max_live_runtime_metadata_bytes",
        ],
    );
    let seal = section(&source, "pub fn seal(", "fn validate_expected_bindings");
    require_all(
        seal,
        &[
            "runtime_pointer_and_schema_metadata_bytes_v3",
            "max_live_runtime_metadata_bytes",
            "admitted_pointer_and_schema_metadata_bytes",
        ],
    );
}

#[test]
fn every_owned_device_buffer_has_stream_ordered_normal_and_failure_drop() {
    let source = runtime_source();
    require_all(
        &source,
        &[
            "struct StreamOrderedDeviceBufferV3",
            "impl<T: DeviceCopy> Drop for StreamOrderedDeviceBufferV3",
        ],
    );

    let ordered_drop = item_block(
        &source,
        "impl<T: DeviceCopy> Drop for StreamOrderedDeviceBufferV3",
    );
    require_all(
        ordered_drop,
        &[
            "CurrentContext::set_current",
            ".drop_async(",
            "std::mem::forget",
        ],
    );
    assert!(!ordered_drop.contains("synchronize("));

    for direct_owned_allocation in [
        "DeviceBuffer::<f64>::uninitialized_async",
        "DeviceBuffer::<u8>::uninitialized_async",
        "DeviceBuffer::<u32>::uninitialized_async",
        "DeviceBuffer::<u64>::uninitialized_async",
        "DeviceBuffer::<f64>::zeroed_async",
        "DeviceBuffer::<u8>::zeroed_async",
        "DeviceBuffer::<u32>::zeroed_async",
        "DeviceBuffer::<u64>::zeroed_async",
    ] {
        assert!(
            !source.contains(direct_owned_allocation),
            "V3 allocation {direct_owned_allocation:?} bypasses the stream-ordered owner"
        );
    }
    assert!(
        !source.contains("Result<(LockedBuffer<T>, DeviceBuffer<T>)"),
        "async compact metadata must return a stream-ordered device owner"
    );

    for start in [
        "struct ResidentAppendTransactionV3",
        "pub struct ResidentFeatureStoreAssemblerV3",
        "struct ResidentHashTransientV3",
        "pub struct ResidentFeatureStoreOwnerV3",
    ] {
        let owned_fields = item_block(&source, start);
        assert!(
            owned_fields.contains("StreamOrderedDeviceBufferV3"),
            "owned V3 allocation section {start:?} bypasses stream-ordered destruction"
        );
        assert!(
            !owned_fields.contains("Option<DeviceBuffer<")
                && !owned_fields.contains(": DeviceBuffer<"),
            "owned V3 allocation section {start:?} can fall back to synchronous DeviceBuffer::drop"
        );
    }
}

#[test]
fn native_v3_rejects_null_default_stream_and_selected_device_grid_overflow() {
    let source = native_source();
    let initialize = section(
        &source,
        "neoethos_resident_initialize_validity_u4_v3(",
        "neoethos_resident_pack_batch_to_bar_major_f64_u4_v3(",
    );
    require_all(
        initialize,
        &[
            "search_bar_major_validity_u4 == nullptr",
            "validity_code_error == nullptr",
            "stream == nullptr",
        ],
    );

    let pack = section(
        &source,
        "neoethos_resident_pack_batch_to_bar_major_f64_u4_v3(",
        "neoethos_resident_canonical_merkle_sha256_v3(",
    );
    require_all(
        pack,
        &[
            "source_addresses == nullptr",
            "source_offsets == nullptr",
            "source_validity_addresses == nullptr",
            "source_validity_offsets == nullptr",
            "search_bar_major_values == nullptr",
            "search_bar_major_validity_u4 == nullptr",
            "validity_code_error == nullptr",
            "stream == nullptr",
            "validate_current_device_grid(grid.x, grid.y)",
        ],
    );
    require_all(
        &source,
        &[
            "cudaGetDevice(&device)",
            "cudaGetDeviceProperties(&properties, device)",
            "properties.maxGridSize[0]",
            "properties.maxGridSize[1]",
        ],
    );

    let merkle_start = position(&source, "neoethos_resident_canonical_merkle_sha256_v3(");
    let merkle = &source[merkle_start..];
    require_all(
        merkle,
        &[
            "timestamps == nullptr",
            "name_offsets == nullptr",
            "name_bytes == nullptr",
            "search_bar_major_values == nullptr",
            "search_bar_major_validity_u4 == nullptr",
            "merkle_scratch_a == nullptr",
            "merkle_scratch_b == nullptr",
            "digest == nullptr",
            "stream == nullptr",
        ],
    );
}

#[test]
fn strict_gpu_search_reaches_v3_resident_bind_without_legacy_full_dataset_upload() {
    let search_root = crate_root().join("../neoethos-search/src");
    let sources = rust_sources_below(&search_root);
    let strict_modules = sources
        .iter()
        .filter(|(_, source)| source.contains("ResidentFeatureStoreImportV3"))
        .collect::<Vec<_>>();
    assert!(
        !strict_modules.is_empty(),
        "strict GPU Search has no V3 resident feature-store consumer"
    );

    let mut resident_bind_is_reachable = false;
    for (path, source) in strict_modules {
        resident_bind_is_reachable |= source.contains("bind_resident_feature_store_v3");
        for forbidden in [
            ".upload_dataset(",
            "neoethos_gpu_cuda_population_upload_dataset",
            "indicators_feature_major",
            "PopulationDataset::new",
        ] {
            assert!(
                !source.contains(forbidden),
                "strict V3 module {} still reaches legacy full staging via {forbidden:?}",
                path.display()
            );
        }
    }
    assert!(
        resident_bind_is_reachable,
        "strict GPU Search never binds the sealed V3 resident store directly"
    );
}
