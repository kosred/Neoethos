from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)

cube_path = Path("crates/neoethos-search/src/cubecl_eval.rs")
cube = cube_path.read_text(encoding="utf-8")
telemetry = '#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]\npub(crate) struct CubeClTransferTelemetry {\n    pub gpu_calls: u64,\n    pub resident_cache_hits: u64,\n    pub resident_cache_misses: u64,\n    pub resident_upload_bytes: u64,\n    pub streamed_dataset_upload_bytes: u64,\n    pub gene_uploads: u64,\n    pub gene_upload_bytes: u64,\n    pub full_readbacks: u64,\n    pub full_readback_bytes: u64,\n    pub compact_readbacks: u64,\n    pub compact_readback_bytes: u64,\n    pub chained_reuploads: u64,\n    pub synchronization_events: u64,\n}\n\nmod transfer_telemetry {\n    use super::CubeClTransferTelemetry;\n    use std::sync::atomic::{AtomicU64, Ordering};\n\n    static GPU_CALLS: AtomicU64 = AtomicU64::new(0);\n    static RESIDENT_CACHE_HITS: AtomicU64 = AtomicU64::new(0);\n    static RESIDENT_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);\n    static RESIDENT_UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);\n    static STREAMED_DATASET_UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);\n    static GENE_UPLOADS: AtomicU64 = AtomicU64::new(0);\n    static GENE_UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);\n    static FULL_READBACKS: AtomicU64 = AtomicU64::new(0);\n    static FULL_READBACK_BYTES: AtomicU64 = AtomicU64::new(0);\n    static COMPACT_READBACKS: AtomicU64 = AtomicU64::new(0);\n    static COMPACT_READBACK_BYTES: AtomicU64 = AtomicU64::new(0);\n    static CHAINED_REUPLOADS: AtomicU64 = AtomicU64::new(0);\n    static SYNCHRONIZATION_EVENTS: AtomicU64 = AtomicU64::new(0);\n\n    pub(super) fn reset() {\n        for counter in [\n            &GPU_CALLS,\n            &RESIDENT_CACHE_HITS,\n            &RESIDENT_CACHE_MISSES,\n            &RESIDENT_UPLOAD_BYTES,\n            &STREAMED_DATASET_UPLOAD_BYTES,\n            &GENE_UPLOADS,\n            &GENE_UPLOAD_BYTES,\n            &FULL_READBACKS,\n            &FULL_READBACK_BYTES,\n            &COMPACT_READBACKS,\n            &COMPACT_READBACK_BYTES,\n            &CHAINED_REUPLOADS,\n            &SYNCHRONIZATION_EVENTS,\n        ] {\n            counter.store(0, Ordering::Relaxed);\n        }\n    }\n\n    pub(super) fn snapshot() -> CubeClTransferTelemetry {\n        CubeClTransferTelemetry {\n            gpu_calls: GPU_CALLS.load(Ordering::Relaxed),\n            resident_cache_hits: RESIDENT_CACHE_HITS.load(Ordering::Relaxed),\n            resident_cache_misses: RESIDENT_CACHE_MISSES.load(Ordering::Relaxed),\n            resident_upload_bytes: RESIDENT_UPLOAD_BYTES.load(Ordering::Relaxed),\n            streamed_dataset_upload_bytes: STREAMED_DATASET_UPLOAD_BYTES.load(Ordering::Relaxed),\n            gene_uploads: GENE_UPLOADS.load(Ordering::Relaxed),\n            gene_upload_bytes: GENE_UPLOAD_BYTES.load(Ordering::Relaxed),\n            full_readbacks: FULL_READBACKS.load(Ordering::Relaxed),\n            full_readback_bytes: FULL_READBACK_BYTES.load(Ordering::Relaxed),\n            compact_readbacks: COMPACT_READBACKS.load(Ordering::Relaxed),\n            compact_readback_bytes: COMPACT_READBACK_BYTES.load(Ordering::Relaxed),\n            chained_reuploads: CHAINED_REUPLOADS.load(Ordering::Relaxed),\n            synchronization_events: SYNCHRONIZATION_EVENTS.load(Ordering::Relaxed),\n        }\n    }\n\n    pub(super) fn record_call() {\n        GPU_CALLS.fetch_add(1, Ordering::Relaxed);\n    }\n\n    pub(super) fn record_streamed_dataset_upload(bytes: usize) {\n        STREAMED_DATASET_UPLOAD_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);\n    }\n\n    pub(super) fn record_gene_upload(bytes: usize) {\n        GENE_UPLOADS.fetch_add(1, Ordering::Relaxed);\n        GENE_UPLOAD_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);\n    }\n\n    pub(super) fn record_resident_hit() {\n        RESIDENT_CACHE_HITS.fetch_add(1, Ordering::Relaxed);\n    }\n\n    pub(super) fn record_resident_miss(bytes: usize) {\n        RESIDENT_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);\n        RESIDENT_UPLOAD_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);\n    }\n\n    pub(super) fn record_readback(use_fused: bool, compact_bytes: usize, dense_bytes: usize) {\n        COMPACT_READBACKS.fetch_add(1, Ordering::Relaxed);\n        COMPACT_READBACK_BYTES.fetch_add(compact_bytes as u64, Ordering::Relaxed);\n        SYNCHRONIZATION_EVENTS.fetch_add(1, Ordering::Relaxed);\n        if !use_fused && dense_bytes > 0 {\n            FULL_READBACKS.fetch_add(1, Ordering::Relaxed);\n            FULL_READBACK_BYTES.fetch_add(dense_bytes as u64, Ordering::Relaxed);\n            CHAINED_REUPLOADS.fetch_add(1, Ordering::Relaxed);\n        }\n    }\n}\n\npub(crate) fn reset_cubecl_transfer_telemetry() {\n    transfer_telemetry::reset();\n}\n\npub(crate) fn cubecl_transfer_telemetry_snapshot() -> CubeClTransferTelemetry {\n    transfer_telemetry::snapshot()\n}\n'
marker = "const FTMO_WIDTH: usize = 6;\n"
cube = replace_once(cube, marker, marker + "\n" + telemetry + "\n", "telemetry module insertion")

cube = replace_once(
    cube,
    '''            let Ok(mut st) = state().lock() else {
                return client.create_from_slice(bytes);
            };
''',
    '''            let Ok(mut st) = state().lock() else {
                super::transfer_telemetry::record_resident_miss(bytes.len());
                return client.create_from_slice(bytes);
            };
''',
    "resident cache poisoned-lock upload",
)
cube = replace_once(
    cube,
    '''            if let Some((_, handle)) = st.map.get(&key) {
                return handle.clone();
            }
''',
    '''            if let Some((_, handle)) = st.map.get(&key) {
                super::transfer_telemetry::record_resident_hit();
                return handle.clone();
            }
''',
    "resident cache hit",
)
cube = replace_once(
    cube,
    '''        // Upload OUTSIDE the lock (can take milliseconds for GB-scale buffers).
        let handle = client.create_from_slice(bytes);
''',
    '''        // Upload OUTSIDE the lock (can take milliseconds for GB-scale buffers).
        let handle = client.create_from_slice(bytes);
        super::transfer_telemetry::record_resident_miss(bytes.len());
''',
    "resident cache miss",
)

validation = '''    if high.len() != n_samples
        || low.len() != n_samples
        || month_idx.len() != n_samples
        || day_idx.len() != n_samples
        || indicators.ncols() != n_samples
        || sl_pips.len() != n_genes
        || tp_pips.len() != n_genes
    {
        bail!("cuda population evaluate path received inconsistent dimensions");
    }
'''
record = validation + '''
    transfer_telemetry::record_call();
'''
cube = replace_once(cube, validation, record, "GPU call telemetry")

cube = replace_once(
    cube,
    '    // Gene-independent inputs are identical every window — upload ONCE.\n    let (\n',
    '    // Gene-independent inputs are identical every window — upload ONCE.\n    let gene_upload_bytes = gene_offsets.len().saturating_mul(std::mem::size_of::<i32>())\n        + gene_indices.len().saturating_mul(std::mem::size_of::<i32>())\n        + gene_weights.len().saturating_mul(std::mem::size_of::<F>())\n        + long_thr.len().saturating_mul(std::mem::size_of::<F>())\n        + short_thr.len().saturating_mul(std::mem::size_of::<F>())\n        + gene_smc_flags_flat.len().saturating_mul(std::mem::size_of::<i32>())\n        + smc_weights.len().saturating_mul(std::mem::size_of::<F>())\n        + sl_pips.len().saturating_mul(std::mem::size_of::<f32>())\n        + tp_pips.len().saturating_mul(std::mem::size_of::<f32>())\n        + stop_vol_mult.len().saturating_mul(std::mem::size_of::<f32>());\n    transfer_telemetry::record_gene_upload(gene_upload_bytes);\n    let (\n',
    "fused gene upload telemetry",
)

cube = replace_once(
    cube,
    '        let smc_window = &smc_data_flat[s0 * SMC_WIDTH..s1 * SMC_WIDTH];\n',
    '        let smc_window = &smc_data_flat[s0 * SMC_WIDTH..s1 * SMC_WIDTH];\n        transfer_telemetry::record_streamed_dataset_upload(\n            ind_window.len().saturating_mul(std::mem::size_of::<F>())\n                + smc_window.len().saturating_mul(std::mem::size_of::<i32>()),\n        );\n',
    "fused streamed dataset telemetry",
)

result_marker = '''    let mut results = Vec::with_capacity(n_genes);
'''
readback = '''    let compact_readback_bytes = metrics_flat
        .len()
        .saturating_mul(std::mem::size_of::<f32>())
        + trade_counts.len().saturating_mul(std::mem::size_of::<i32>())
        + monthly_flat.len().saturating_mul(std::mem::size_of::<f32>())
        + month_counts.len().saturating_mul(std::mem::size_of::<i32>())
        + month_start_eq_flat
            .len()
            .saturating_mul(std::mem::size_of::<f32>());
    let dense_readback_bytes = n_genes
        .saturating_mul(n_samples)
        .saturating_mul(std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
    transfer_telemetry::record_readback(use_fused, compact_readback_bytes, dense_readback_bytes);

''' + result_marker
first = cube.find(result_marker, cube.index("pub(crate) fn try_evaluate_population_cuda"))
if first == -1:
    raise RuntimeError("metrics result assembly marker missing")
cube = cube[:first] + readback + cube[first + len(result_marker):]
cube_path.write_text(cube, encoding="utf-8")

mod_path = Path("crates/neoethos-search/src/gpu_native/mod.rs")
mods = mod_path.read_text(encoding="utf-8")
mods = replace_once(
    mods,
    "pub mod parity_hierarchy;\n",
    "pub mod parity_hierarchy;\npub mod prototype_a;\n",
    "prototype A module export",
)
mod_path.write_text(mods, encoding="utf-8")
print("instrumented CubeCL transfer/cache behaviour for Prototype A evidence")
