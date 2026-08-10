//! Host mirrors of the two CUDA-specific mechanics inside Prototype B.
//!
//! Prototype C already proves the shared population *semantics* on a real
//! adapter. What is unique to Prototype B, and therefore unproven until a CUDA
//! device exists, is how it parallelises that work:
//!
//! 1. a warp searches one event's bar window lane-strided, then reduces to the
//!    earliest hit with `__shfl_down_sync`;
//! 2. a block emits one candidate's causal entries using a chunked
//!    Hillis-Steele scan so slots stay in canonical bar order.
//!
//! Both are transcribed here as plain Rust and checked against straightforward
//! serial references. A mirror is not the kernel: it cannot catch a CUDA
//! syntax error, a race or a precision difference. It does catch the expensive
//! class of bug — an algorithm that is simply wrong — before a single paid GPU
//! minute is spent. The kernel and the mirror must be edited together.
//!
//! ⚠ NEITHER MECHANIC IS STILL IN THE KERNEL.
//!
//! `population_first_hit_kernel` and `population_emit_events_kernel` were
//! deleted from `prototype_b_population.cu` when signal synthesis was fused
//! into the walk. Exit detection now happens inline against the position the
//! walk is holding, and entries are opened straight from the synthesised
//! signal, so there is no event stream for a warp to search and no scan to
//! order slots with. What is below is therefore a HISTORICAL reference: its
//! tests still pass and still prove those two algorithms, but they no longer
//! guard anything that runs.
//!
//! It is left in place rather than deleted in the same change that rewrote the
//! kernel, because deleting a proof and rewriting the thing it proved at once
//! is how a regression gets through. Delete it once the fused walk has passed
//! the 147/147 parity suite on a real card.

/// Exit reasons, mirroring the kernel's encoding.
pub const MIRROR_EXIT_NONE: i32 = 0;
pub const MIRROR_EXIT_STOP: i32 = 1;
pub const MIRROR_EXIT_TARGET: i32 = 2;
pub const MIRROR_EXIT_MAX_HOLD: i32 = 3;
pub const MIRROR_EXIT_GAP: i32 = 4;

/// Within-bar precedence: a gap drains every active event before level checks,
/// and the max-hold sweep only fires for events still active afterwards.
pub fn exit_priority(reason: i32) -> i32 {
    match reason {
        MIRROR_EXIT_GAP => 0,
        MIRROR_EXIT_STOP => 1,
        MIRROR_EXIT_TARGET => 2,
        MIRROR_EXIT_MAX_HOLD => 3,
        _ => i32::MAX,
    }
}

/// Serial reference: scan the window forward and stop at the first bar that
/// produces any reason, breaking same-bar ties by priority.
pub fn serial_first_hit(reasons: &[i32]) -> Option<(usize, i32)> {
    reasons
        .iter()
        .enumerate()
        .find(|(_, reason)| **reason != MIRROR_EXIT_NONE)
        .map(|(bar, reason)| (bar, *reason))
}

/// Mirror of `population_first_hit_kernel`: lane-strided scan plus the
/// `__shfl_down_sync` tree reduction, including the `INT_MAX` sentinel and the
/// lexicographic `(bar, priority)` comparison.
pub fn warp_first_hit(reasons: &[i32], warp_size: usize) -> Option<(usize, i32)> {
    assert!(
        warp_size.is_power_of_two(),
        "a warp width must be a power of two"
    );

    // Per-lane strided scan, exactly as each lane runs it on device.
    let mut best_bar = vec![i32::MAX; warp_size];
    let mut best_priority = vec![i32::MAX; warp_size];
    let mut best_reason = vec![MIRROR_EXIT_NONE; warp_size];
    for (lane, (bar_slot, priority_slot)) in best_bar
        .iter_mut()
        .zip(best_priority.iter_mut())
        .enumerate()
    {
        let mut bar = lane;
        while bar < reasons.len() {
            let reason = reasons[bar];
            if reason != MIRROR_EXIT_NONE {
                let priority = exit_priority(reason);
                let bar_index = bar as i32;
                if bar_index < *bar_slot || (bar_index == *bar_slot && priority < *priority_slot) {
                    *bar_slot = bar_index;
                    *priority_slot = priority;
                    best_reason[lane] = reason;
                }
            }
            bar += warp_size;
        }
    }

    // Tree reduction. Every lane participates at every step, matching the full
    // 0xffffffff mask the kernel uses.
    let mut offset = warp_size / 2;
    while offset > 0 {
        for lane in 0..warp_size {
            let partner = lane + offset;
            let (other_bar, other_priority, other_reason) = if partner < warp_size {
                (
                    best_bar[partner],
                    best_priority[partner],
                    best_reason[partner],
                )
            } else {
                (i32::MAX, i32::MAX, MIRROR_EXIT_NONE)
            };
            if other_bar < best_bar[lane]
                || (other_bar == best_bar[lane] && other_priority < best_priority[lane])
            {
                best_bar[lane] = other_bar;
                best_priority[lane] = other_priority;
                best_reason[lane] = other_reason;
            }
        }
        offset /= 2;
    }

    (best_bar[0] != i32::MAX).then(|| (best_bar[0] as usize, best_reason[0]))
}

/// Mirror of the emission block scan in `population_emit_events_kernel`:
/// chunked inclusive Hillis-Steele scan converted to an exclusive offset, with
/// a running per-block base.
pub fn block_scan_emission_slots(flags: &[bool], block: usize) -> Vec<usize> {
    assert!(block > 0, "a block must contain at least one thread");
    let mut slots = Vec::new();
    let mut chunk_base = 0_usize;
    let mut chunk_start = 0_usize;
    while chunk_start < flags.len() {
        let mut scan: Vec<usize> = (0..block)
            .map(|thread| {
                let index = chunk_start + thread;
                usize::from(flags.get(index).copied().unwrap_or(false))
            })
            .collect();

        let mut stride = 1;
        while stride < block {
            let previous = scan.clone();
            for thread in stride..block {
                scan[thread] += previous[thread - stride];
            }
            stride <<= 1;
        }

        for thread in 0..block {
            let index = chunk_start + thread;
            if flags.get(index).copied().unwrap_or(false) {
                let exclusive = scan[thread] - 1;
                slots.push(chunk_base + exclusive);
            }
        }
        chunk_base += scan[block - 1];
        chunk_start += block;
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic, seeded pseudo-random stream: the mirrors must be checked
    /// over many shapes, and a fixed seed keeps a failure reproducible.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() as usize) % bound.max(1)
        }
    }

    fn random_reasons(rng: &mut Rng, len: usize, density: usize) -> Vec<i32> {
        (0..len)
            .map(|_| {
                if rng.below(100) < density {
                    match rng.below(4) {
                        0 => MIRROR_EXIT_STOP,
                        1 => MIRROR_EXIT_TARGET,
                        2 => MIRROR_EXIT_MAX_HOLD,
                        _ => MIRROR_EXIT_GAP,
                    }
                } else {
                    MIRROR_EXIT_NONE
                }
            })
            .collect()
    }

    #[test]
    fn warp_reduction_matches_the_serial_scan_across_widths_and_densities() {
        let mut rng = Rng(0x5eed_1234);
        for warp_size in [1_usize, 2, 4, 8, 16, 32] {
            for density in [0_usize, 1, 5, 25, 100] {
                for _ in 0..200 {
                    let len = rng.below(200);
                    let reasons = random_reasons(&mut rng, len, density);
                    assert_eq!(
                        warp_first_hit(&reasons, warp_size),
                        serial_first_hit(&reasons),
                        "warp={warp_size} density={density} reasons={reasons:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn an_empty_window_reduces_to_no_hit() {
        for warp_size in [1_usize, 2, 32] {
            assert_eq!(warp_first_hit(&[], warp_size), None);
            assert_eq!(warp_first_hit(&[MIRROR_EXIT_NONE; 7], warp_size), None);
        }
    }

    #[test]
    fn a_hit_in_the_last_lane_of_the_last_stride_still_wins() {
        // The only non-zero bar sits where a naive reduction that ignores the
        // sentinel would drop it.
        let mut reasons = vec![MIRROR_EXIT_NONE; 64];
        reasons[63] = MIRROR_EXIT_TARGET;
        assert_eq!(warp_first_hit(&reasons, 32), Some((63, MIRROR_EXIT_TARGET)));
    }

    #[test]
    fn the_earliest_bar_wins_even_when_a_later_bar_has_higher_priority() {
        // A gap at bar 9 must not beat a target at bar 2: bar order dominates,
        // priority only breaks ties inside one bar.
        let mut reasons = vec![MIRROR_EXIT_NONE; 32];
        reasons[2] = MIRROR_EXIT_TARGET;
        reasons[9] = MIRROR_EXIT_GAP;
        assert_eq!(warp_first_hit(&reasons, 32), Some((2, MIRROR_EXIT_TARGET)));
        assert_eq!(warp_first_hit(&reasons, 4), Some((2, MIRROR_EXIT_TARGET)));
    }

    #[test]
    fn every_lane_stride_offset_is_covered() {
        // One hit at each position in turn: any lane whose stride is skipped
        // would surface here.
        for bar in 0..97_usize {
            let mut reasons = vec![MIRROR_EXIT_NONE; 97];
            reasons[bar] = MIRROR_EXIT_STOP;
            for warp_size in [1_usize, 2, 8, 32] {
                assert_eq!(
                    warp_first_hit(&reasons, warp_size),
                    Some((bar, MIRROR_EXIT_STOP)),
                    "bar {bar} lost at warp width {warp_size}"
                );
            }
        }
    }

    #[test]
    fn emission_scan_produces_dense_canonical_slots() {
        let mut rng = Rng(0xfeed_9876);
        for block in [1_usize, 2, 4, 8, 256] {
            for _ in 0..200 {
                let len = rng.below(1000);
                let flags: Vec<bool> = (0..len).map(|_| rng.below(100) < 30).collect();
                let slots = block_scan_emission_slots(&flags, block);
                let expected: Vec<usize> =
                    (0..flags.iter().filter(|flag| **flag).count()).collect();
                assert_eq!(slots, expected, "block={block} flags={flags:?}");
            }
        }
    }

    #[test]
    fn emission_slots_follow_bar_order_across_chunk_boundaries() {
        // Signals straddling a chunk boundary must keep ascending order: this
        // is what makes the event stream candidate-major and bar-ascending.
        let block = 4;
        let flags = vec![
            false, true, false, true, // chunk 0 -> slots 0,1
            true, false, false, false, // chunk 1 -> slot 2
            false, false, true, true, // chunk 2 -> slots 3,4
        ];
        assert_eq!(
            block_scan_emission_slots(&flags, block),
            vec![0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn an_all_signal_population_emits_one_slot_per_bar() {
        let flags = vec![true; 300];
        let slots = block_scan_emission_slots(&flags, 256);
        assert_eq!(slots.len(), 300);
        assert!(slots.iter().enumerate().all(|(index, slot)| index == *slot));
    }
}
