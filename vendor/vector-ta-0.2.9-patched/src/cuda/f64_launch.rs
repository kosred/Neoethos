//! Launch planning shared by the from-scratch f64 kernels.
//!
//! # The shape these kernels have, and why they need a planner
//!
//! Nine of the indicators in this crate shipped as one-line empty CUDA stubs,
//! and eight more had no `.cu` file at all. Every one of them is either a
//! serial recurrence over bars (a streaming state machine, an EMA cascade) or a
//! per-window computation with a working set far larger than a register file.
//! The correct CUDA shape for both is ONE THREAD PER PARAMETER ROW walking bars
//! in ascending order — the CPU's accumulation order, preserved exactly — with
//! that thread's working set in global memory.
//!
//! That working set is the problem this module solves. A row of
//! `goertzel_cycle_composite_wave` needs roughly 4,700 doubles of scratch; a
//! sweep of 4,096 rows would demand 154 GB if every row got its own slab at
//! once. So the kernels are launched with a bounded number of SLOTS and each
//! slot loops over the rows assigned to it. [`plan_slots`] chooses that number
//! from the memory the card actually has free.
//!
//! # NEVER-OOM
//!
//! Peak memory here is a function of the AVAILABLE HARDWARE, never of a
//! parameter the operator typed. A wider sweep runs SLOWER on a small card, in
//! more passes; it does not run out of memory. That is the whole point of
//! planning slots instead of allocating per row.

#![cfg(feature = "cuda-build-native")]

use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::mem_get_info;

/// VRAM deliberately left unclaimed so a concurrent allocation on the same
/// context does not fail because we took the last byte.
pub const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

/// Threads per block for the row-walking kernels.
///
/// These kernels are memory-bound and highly divergent — each thread walks its
/// own bar series through its own scratch slab — so occupancy is not the lever
/// and a small block keeps the per-SM scratch working set inside cache.
pub const ROW_BLOCK_X: u32 = 64;

/// Why a launch could not be planned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchPlanError {
    /// Not even ONE slot fits. The card is too small for a single row of this
    /// indicator at this series length; there is no smaller unit to fall back
    /// to, so this is a hard error rather than a silent host computation.
    NoRoomForOneSlot {
        indicator: &'static str,
        bytes_per_slot: usize,
        fixed_bytes: usize,
        free: usize,
        headroom: usize,
    },
    /// An intermediate byte count overflowed `usize`.
    SizeOverflow {
        indicator: &'static str,
        what: &'static str,
    },
}

impl std::fmt::Display for LaunchPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaunchPlanError::NoRoomForOneSlot {
                indicator,
                bytes_per_slot,
                fixed_bytes,
                free,
                headroom,
            } => write!(
                f,
                "{indicator}: no room on the device for a single row — \
                 scratch/row={bytes_per_slot} fixed={fixed_bytes} free={free} headroom={headroom}"
            ),
            LaunchPlanError::SizeOverflow { indicator, what } => {
                write!(f, "{indicator}: size overflow computing {what}")
            }
        }
    }
}

impl std::error::Error for LaunchPlanError {}

/// How many rows a launch will process concurrently, and the grid that does it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchPlan {
    /// Concurrent scratch slots. Each device thread `t < slots` walks rows
    /// `t, t + slots, t + 2*slots, …`, so the kernel is correct for any value
    /// here from 1 upwards and only the runtime changes.
    pub slots: usize,
    pub grid: GridSize,
    pub block: BlockSize,
}

/// Choose the slot count from free VRAM.
///
/// `fixed_bytes` is everything allocated once per launch regardless of slot
/// count — inputs, per-row parameter vectors, output matrices. `bytes_per_slot`
/// is the per-row scratch. Both are computed by the caller because only the
/// caller knows the kernel's scratch layout.
///
/// `mem_get_info` failing is treated as "assume one slot", not as "assume
/// unlimited": a wrong guess upwards is an out-of-memory crash, a wrong guess
/// downwards is a slow run.
pub fn plan_slots(
    indicator: &'static str,
    rows: usize,
    fixed_bytes: usize,
    bytes_per_slot: usize,
    headroom: usize,
) -> Result<LaunchPlan, LaunchPlanError> {
    let rows = rows.max(1);

    // A kernel with no per-row scratch can run every row at once.
    if bytes_per_slot == 0 {
        return Ok(plan_for(rows));
    }

    let (free, _total) = mem_get_info().unwrap_or((0, 0));
    let budget = free.saturating_sub(headroom).saturating_sub(fixed_bytes);

    let affordable = budget / bytes_per_slot;
    if affordable == 0 {
        // `free == 0` means `mem_get_info` failed rather than that the card is
        // literally full; in that case one slot is the honest floor and the
        // allocation itself will report a real out-of-memory if it is wrong.
        if free == 0 {
            return Ok(plan_for(1));
        }
        return Err(LaunchPlanError::NoRoomForOneSlot {
            indicator,
            bytes_per_slot,
            fixed_bytes,
            free,
            headroom,
        });
    }

    Ok(plan_for(affordable.min(rows)))
}

fn plan_for(slots: usize) -> LaunchPlan {
    let slots = slots.max(1);
    let grid_x = ((slots as u64 + ROW_BLOCK_X as u64 - 1) / ROW_BLOCK_X as u64).max(1);
    LaunchPlan {
        slots,
        grid: GridSize::x(grid_x as u32),
        block: BlockSize::x(ROW_BLOCK_X),
    }
}

/// Total scratch elements for a plan: `slots * per_slot`, checked.
pub fn scratch_elems(
    indicator: &'static str,
    what: &'static str,
    slots: usize,
    per_slot: usize,
) -> Result<usize, LaunchPlanError> {
    slots
        .checked_mul(per_slot)
        .ok_or(LaunchPlanError::SizeOverflow { indicator, what })
}

/// `a * b`, checked, with the indicator and the quantity named in the error.
pub fn checked_mul(
    indicator: &'static str,
    what: &'static str,
    a: usize,
    b: usize,
) -> Result<usize, LaunchPlanError> {
    a.checked_mul(b)
        .ok_or(LaunchPlanError::SizeOverflow { indicator, what })
}

/// Reject a grid/block the device cannot run, BEFORE launching it, so the
/// failure names the configuration instead of surfacing as a generic
/// `InvalidConfiguration` from the driver.
pub fn validate_launch(
    device_id: u32,
    grid: GridSize,
    block: BlockSize,
) -> Result<(), cust::error::CudaError> {
    let device = Device::get_device(device_id)?;
    let max_threads = device
        .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
        .unwrap_or(1024) as u32;
    let max_grid_x = device
        .get_attribute(DeviceAttribute::MaxGridDimX)
        .unwrap_or(i32::MAX) as u32;
    let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
    if threads > max_threads || grid.x > max_grid_x {
        return Err(cust::error::CudaError::InvalidValue);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_scratch_runs_every_row_at_once() {
        let plan = plan_slots("t", 5000, 0, 0, DEFAULT_HEADROOM).expect("plan");
        assert_eq!(plan.slots, 5000);
        assert_eq!(plan.block.x, ROW_BLOCK_X);
        assert_eq!(plan.grid.x, (5000 + 63) / 64);
    }

    #[test]
    fn slots_never_exceed_rows() {
        // With no device present `mem_get_info` fails, which the planner reads
        // as "one slot" rather than as "unlimited".
        let plan = plan_slots("t", 3, 0, 1024, DEFAULT_HEADROOM).expect("plan");
        assert!(plan.slots >= 1 && plan.slots <= 3);
    }

    #[test]
    fn overflow_is_named_not_wrapped() {
        let err = checked_mul("t", "rows*cols", usize::MAX, 2).unwrap_err();
        assert_eq!(
            err,
            LaunchPlanError::SizeOverflow {
                indicator: "t",
                what: "rows*cols"
            }
        );
    }
}
