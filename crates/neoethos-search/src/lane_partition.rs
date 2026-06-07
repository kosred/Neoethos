//! CSR population partitioning for multi-GPU sharding (scheduler Stage 2).
//!
//! The GA stores a population's gene→indicator mapping in CSR form:
//! `gene_offsets` (length `n_genes + 1`) plus flat `gene_indices` /
//! `gene_weights`, where gene `g` owns `indices[offsets[g]..offsets[g+1]]`. To
//! run the GPU lane across SEVERAL cards, the GPU-assigned gene prefix
//! `[0, gpu_count)` is split into contiguous per-device chunks. Each device's
//! kernel expects a **0-based** CSR, so this module rebases each chunk's
//! offsets and reports the flat index range to slice for that chunk.
//!
//! CSR rebasing is the bug-prone part of multi-GPU sharding — an off-by-one
//! silently corrupts fitness — so it lives here as a pure, GPU-free function
//! that is exhaustively unit-tested. The device-execution glue that consumes
//! these partitions lives in `eval.rs` behind `feature = "gpu"` (hence the
//! allow below until that wiring lands).
#![allow(dead_code)]

/// One GPU device's contiguous slice of the population, CSR-rebased to 0.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LanePartition {
    /// The concrete device id this chunk runs on (from the caller's list).
    pub device_id: usize,
    /// Gene range `[gene_start, gene_end)` within the original population.
    pub gene_start: usize,
    pub gene_end: usize,
    /// Rebased offsets (length `gene_end - gene_start + 1`); `[0]` is always 0.
    pub rebased_offsets: Vec<i32>,
    /// Flat range `[idx_start, idx_end)` into `gene_indices` / `gene_weights`.
    pub idx_start: usize,
    pub idx_end: usize,
}

impl LanePartition {
    pub fn gene_count(&self) -> usize {
        self.gene_end - self.gene_start
    }
}

/// Partition the GPU-assigned gene prefix `[0, gpu_count)` across `device_ids`,
/// giving each device at most `genes_per_device_cap` genes (its VRAM cap).
///
/// Genes are assigned contiguously and as evenly as possible. If
/// `gpu_count > device_ids.len() * genes_per_device_cap`, the surplus genes are
/// NOT placed here — the caller must route `[placed_genes(..), gpu_count)` to
/// the CPU lane. Surfacing that boundary (rather than silently truncating)
/// keeps every gene accounted for.
///
/// Returns one [`LanePartition`] per device that received `>= 1` gene.
///
/// `gene_offsets` must be the FULL population offsets (length `n_genes + 1`)
/// with `gpu_count <= n_genes`; only `[0..=gpu_count]` is read.
pub(crate) fn partition_gpu_lanes(
    gene_offsets: &[i32],
    gpu_count: usize,
    device_ids: &[usize],
    genes_per_device_cap: usize,
) -> Vec<LanePartition> {
    let n_dev = device_ids.len();
    if n_dev == 0 || gpu_count == 0 || genes_per_device_cap == 0 {
        return Vec::new();
    }
    debug_assert!(
        gene_offsets.len() > gpu_count,
        "gene_offsets must have at least gpu_count+1 entries"
    );

    // Genes we can actually hold across all cards (the rest go to the CPU lane).
    let placeable = gpu_count.min(n_dev.saturating_mul(genes_per_device_cap));
    if placeable == 0 {
        return Vec::new();
    }

    // Even contiguous split. `base <= cap` and `base + 1 <= cap` always hold
    // here because `placeable <= n_dev * cap` (proven: base == cap only when
    // placeable == n_dev*cap, in which case rem == 0), so the per-device count
    // never needs truncating — but we clamp defensively anyway.
    let base = placeable / n_dev;
    let rem = placeable % n_dev;

    let mut lanes = Vec::with_capacity(n_dev);
    let mut cursor = 0usize;
    for (slot, &device_id) in device_ids.iter().enumerate() {
        if cursor >= placeable {
            break;
        }
        let count = (base + usize::from(slot < rem)).min(genes_per_device_cap);
        if count == 0 {
            continue;
        }
        let gene_start = cursor;
        let gene_end = (cursor + count).min(placeable);
        let base_off = gene_offsets[gene_start];
        let rebased_offsets: Vec<i32> = gene_offsets[gene_start..=gene_end]
            .iter()
            .map(|&o| o - base_off)
            .collect();
        lanes.push(LanePartition {
            device_id,
            gene_start,
            gene_end,
            rebased_offsets,
            idx_start: gene_offsets[gene_start] as usize,
            idx_end: gene_offsets[gene_end] as usize,
        });
        cursor = gene_end;
    }
    lanes
}

/// Total genes placed on the GPU lanes. `[placed, gpu_count)` is the CPU lane.
pub(crate) fn placed_genes(lanes: &[LanePartition]) -> usize {
    lanes.iter().map(LanePartition::gene_count).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_device_one_lane_is_the_prefix_unchanged() {
        let offsets = [0, 2, 4, 6, 8]; // 4 genes, 2 entries each
        let lanes = partition_gpu_lanes(&offsets, 4, &[0], 100);
        assert_eq!(lanes.len(), 1);
        let l = &lanes[0];
        assert_eq!((l.gene_start, l.gene_end), (0, 4));
        assert_eq!(l.rebased_offsets, vec![0, 2, 4, 6, 8]);
        assert_eq!((l.idx_start, l.idx_end), (0, 8));
        assert_eq!(placed_genes(&lanes), 4);
    }

    #[test]
    fn two_devices_split_evenly_and_rebase_each_to_zero() {
        let offsets = [0, 2, 4, 6, 8];
        let lanes = partition_gpu_lanes(&offsets, 4, &[0, 1], 100);
        assert_eq!(lanes.len(), 2);
        assert_eq!((lanes[0].gene_start, lanes[0].gene_end), (0, 2));
        assert_eq!(lanes[0].rebased_offsets, vec![0, 2, 4]);
        assert_eq!((lanes[0].idx_start, lanes[0].idx_end), (0, 4));
        assert_eq!((lanes[1].gene_start, lanes[1].gene_end), (2, 4));
        assert_eq!(lanes[1].rebased_offsets, vec![0, 2, 4]); // rebased from [4,6,8]
        assert_eq!((lanes[1].idx_start, lanes[1].idx_end), (4, 8));
        assert_eq!(placed_genes(&lanes), 4);
    }

    #[test]
    fn uneven_split_gives_the_remainder_to_earlier_devices() {
        let offsets = [0, 1, 2, 3, 4, 5]; // 5 genes, 1 entry each
        let lanes = partition_gpu_lanes(&offsets, 5, &[0, 1], 100);
        assert_eq!(lanes.len(), 2);
        assert_eq!(lanes[0].gene_count(), 3); // base 2 + remainder 1
        assert_eq!(lanes[1].gene_count(), 2);
        assert_eq!(placed_genes(&lanes), 5);
    }

    #[test]
    fn vram_cap_limits_placement_and_surfaces_the_cpu_remainder() {
        let offsets = [0, 1, 2, 3, 4, 5, 6]; // 6 genes
        // 2 devices × cap 2 = 4 placeable; genes 4,5 belong to the CPU lane.
        let lanes = partition_gpu_lanes(&offsets, 6, &[0, 1], 2);
        assert_eq!(placed_genes(&lanes), 4);
        assert!(lanes.iter().all(|l| l.gene_count() <= 2));
        // The caller routes [4, 6) to CPU — nothing is dropped.
        assert_eq!(6 - placed_genes(&lanes), 2);
    }

    #[test]
    fn rebasing_reconstructs_every_gene_exactly() {
        // Varied CSR incl. an empty gene (g1). 5 genes, 10 flat entries.
        let offsets = [0, 3, 3, 7, 9, 10];
        let indices: Vec<i32> = (0..10).collect();
        let device_ids = [4, 7, 9]; // non-contiguous ids carried through
        let lanes = partition_gpu_lanes(&offsets, 5, &device_ids, 100);
        assert_eq!(placed_genes(&lanes), 5);

        for lane in &lanes {
            let lane_idx = &indices[lane.idx_start..lane.idx_end];
            for g in lane.gene_start..lane.gene_end {
                // Original entries for gene g.
                let want = &indices[offsets[g] as usize..offsets[g + 1] as usize];
                // Reconstructed from the rebased lane.
                let k = g - lane.gene_start;
                let a = lane.rebased_offsets[k] as usize;
                let b = lane.rebased_offsets[k + 1] as usize;
                let got = &lane_idx[a..b];
                assert_eq!(got, want, "gene {g} mismatch on device slot {}", lane.device_id);
            }
        }
        // device ids carried through in order.
        assert_eq!(
            lanes.iter().map(|l| l.device_id).collect::<Vec<_>>(),
            vec![4, 7, 9]
        );
    }

    #[test]
    fn degenerate_inputs_yield_no_lanes() {
        let offsets = [0, 1, 2];
        assert!(partition_gpu_lanes(&offsets, 0, &[0], 100).is_empty());
        assert!(partition_gpu_lanes(&offsets, 2, &[], 100).is_empty());
        assert!(partition_gpu_lanes(&offsets, 2, &[0], 0).is_empty());
    }

    #[test]
    fn more_devices_than_genes_leaves_idle_devices_out() {
        let offsets = [0, 1, 2]; // 2 genes
        let lanes = partition_gpu_lanes(&offsets, 2, &[0, 1, 2, 3], 100);
        // Only as many lanes as there are genes (1 gene each for the first two).
        assert_eq!(placed_genes(&lanes), 2);
        assert!(lanes.len() <= 2);
        assert!(lanes.iter().all(|l| l.gene_count() >= 1));
    }
}
