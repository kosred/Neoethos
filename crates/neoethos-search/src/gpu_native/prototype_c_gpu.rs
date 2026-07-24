//! Minimal executable GPU path for Prototype C's compact-event first-hit subset.
//!
//! One GPU invocation consumes an already-compacted event batch, searches the
//! fixed/adaptive-at-entry stop/target path, and returns compact exit bar/reason
//! pairs. It is a correctness prototype, not the final range-index architecture.

use anyhow::{Context, Result, bail};
#[cfg(feature = "gpu-cuda")]
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
use cubecl::wgpu::WgpuRuntime;

use super::prototype_bc::{
    EntryEvent, ExitReason, FirstHit, PositionDirection, PricePath, SameBarPrecedence,
    SparseOutcome,
};

#[cube(launch)]
fn sparse_first_hit_kernel(
    highs: &Array<f32>,
    lows: &Array<f32>,
    entry_bars: &Array<i32>,
    last_bars: &Array<i32>,
    directions: &Array<i32>,
    stop_prices: &Array<f32>,
    target_prices: &Array<f32>,
    precedence: &Array<i32>,
    exit_bars: &mut Array<i32>,
    exit_reasons: &mut Array<i32>,
    event_count: u32,
) {
    if ABSOLUTE_POS < event_count as usize {
        let event = ABSOLUTE_POS;
        let first = entry_bars[event] as usize + 1;
        let last = last_bars[event] as usize;
        let found_bar = RuntimeCell::<i32>::new(-1);
        let found_reason = RuntimeCell::<i32>::new(0);

        for bar in first..last + 1 {
            if found_bar.read() < 0 {
                let high = highs[bar];
                let low = lows[bar];
                let stop_hit = RuntimeCell::<bool>::new(false);
                let target_hit = RuntimeCell::<bool>::new(false);
                if directions[event] > 0 {
                    if low <= stop_prices[event] {
                        stop_hit.store(true);
                    }
                    if high >= target_prices[event] {
                        target_hit.store(true);
                    }
                } else {
                    if high >= stop_prices[event] {
                        stop_hit.store(true);
                    }
                    if low <= target_prices[event] {
                        target_hit.store(true);
                    }
                }

                if stop_hit.read() && target_hit.read() {
                    found_bar.store(bar as i32);
                    if precedence[event] == 0 {
                        found_reason.store(1);
                    } else {
                        found_reason.store(2);
                    }
                } else if stop_hit.read() {
                    found_bar.store(bar as i32);
                    found_reason.store(1);
                } else if target_hit.read() {
                    found_bar.store(bar as i32);
                    found_reason.store(2);
                }
            }
        }

        exit_bars[event] = found_bar.read();
        exit_reasons[event] = found_reason.read();
    }
}

#[cfg(feature = "gpu-cuda")]
pub fn try_prototype_c_gpu_first_hit(
    path: PricePath<'_>,
    events: &[EntryEvent],
) -> Result<Vec<SparseOutcome>> {
    let client = crate::cubecl_eval::create_gpu_client(None)?;
    launch_sparse_first_hit::<CudaRuntime>(&client, path, events)
}

#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
pub fn try_prototype_c_gpu_first_hit(
    path: PricePath<'_>,
    events: &[EntryEvent],
) -> Result<Vec<SparseOutcome>> {
    let client = crate::cubecl_eval::create_gpu_client(None)?;
    launch_sparse_first_hit::<WgpuRuntime>(&client, path, events)
}

fn launch_sparse_first_hit<R: Runtime>(
    client: &ComputeClient<R>,
    path: PricePath<'_>,
    events: &[EntryEvent],
) -> Result<Vec<SparseOutcome>> {
    if path.highs.is_empty() || path.highs.len() != path.lows.len() {
        bail!("Prototype C GPU path requires equal non-empty high/low arrays");
    }
    if events.is_empty() {
        return Ok(Vec::new());
    }

    let rows = path.highs.len();
    let highs: Vec<f32> = path.highs.iter().map(|value| *value as f32).collect();
    let lows: Vec<f32> = path.lows.iter().map(|value| *value as f32).collect();
    if highs.iter().chain(&lows).any(|value| !value.is_finite()) {
        bail!("Prototype C GPU path rejects non-finite prices");
    }

    let mut entry_bars = Vec::with_capacity(events.len());
    let mut last_bars = Vec::with_capacity(events.len());
    let mut directions = Vec::with_capacity(events.len());
    let mut stops = Vec::with_capacity(events.len());
    let mut targets = Vec::with_capacity(events.len());
    let mut precedence = Vec::with_capacity(events.len());
    for event in events {
        let request = event.request;
        if request.entry_bar >= request.last_bar || request.last_bar >= rows {
            bail!("Prototype C GPU event has an invalid bar window");
        }
        if !request.stop_price.is_finite() || !request.target_price.is_finite() {
            bail!("Prototype C GPU event has a non-finite stop/target");
        }
        entry_bars.push(request.entry_bar as i32);
        last_bars.push(request.last_bar as i32);
        directions.push(match request.direction {
            PositionDirection::Long => 1,
            PositionDirection::Short => -1,
        });
        stops.push(request.stop_price as f32);
        targets.push(request.target_price as f32);
        precedence.push(match request.same_bar_precedence {
            SameBarPrecedence::StopFirst => 0,
            SameBarPrecedence::TargetFirst => 1,
        });
    }

    let highs_handle = client.create_from_slice(f32::as_bytes(&highs));
    let lows_handle = client.create_from_slice(f32::as_bytes(&lows));
    let entry_handle = client.create_from_slice(i32::as_bytes(&entry_bars));
    let last_handle = client.create_from_slice(i32::as_bytes(&last_bars));
    let direction_handle = client.create_from_slice(i32::as_bytes(&directions));
    let stop_handle = client.create_from_slice(f32::as_bytes(&stops));
    let target_handle = client.create_from_slice(f32::as_bytes(&targets));
    let precedence_handle = client.create_from_slice(i32::as_bytes(&precedence));
    let exit_bar_handle = client.empty(events.len() * core::mem::size_of::<i32>());
    let exit_reason_handle = client.empty(events.len() * core::mem::size_of::<i32>());

    let units = client.properties().hardware.max_units_per_cube.clamp(1, 64);
    let cubes = (events.len() as u32).div_ceil(units);
    sparse_first_hit_kernel::launch::<R>(
        client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts(highs_handle, highs.len()) },
        unsafe { ArrayArg::from_raw_parts(lows_handle, lows.len()) },
        unsafe { ArrayArg::from_raw_parts(entry_handle, events.len()) },
        unsafe { ArrayArg::from_raw_parts(last_handle, events.len()) },
        unsafe { ArrayArg::from_raw_parts(direction_handle, events.len()) },
        unsafe { ArrayArg::from_raw_parts(stop_handle, events.len()) },
        unsafe { ArrayArg::from_raw_parts(target_handle, events.len()) },
        unsafe { ArrayArg::from_raw_parts(precedence_handle, events.len()) },
        unsafe { ArrayArg::from_raw_parts(exit_bar_handle.clone(), events.len()) },
        unsafe { ArrayArg::from_raw_parts(exit_reason_handle.clone(), events.len()) },
        events.len() as u32,
    );

    let exit_bar_bytes = client
        .read_one(exit_bar_handle)
        .context("read Prototype C exit bars")?;
    let exit_reason_bytes = client
        .read_one(exit_reason_handle)
        .context("read Prototype C exit reasons")?;
    let exit_bars = i32::from_bytes(exit_bar_bytes.as_ref()).to_vec();
    let exit_reasons = i32::from_bytes(exit_reason_bytes.as_ref()).to_vec();
    if exit_bars.len() != events.len() || exit_reasons.len() != events.len() {
        bail!("Prototype C GPU readback shape mismatch");
    }

    events
        .iter()
        .enumerate()
        .map(|(index, event)| {
            let hit = match (exit_bars[index], exit_reasons[index]) {
                (-1, 0) => None,
                (bar, 1) if bar >= 0 => Some(FirstHit {
                    exit_bar: bar as usize,
                    reason: ExitReason::StopLoss,
                }),
                (bar, 2) if bar >= 0 => Some(FirstHit {
                    exit_bar: bar as usize,
                    reason: ExitReason::TakeProfit,
                }),
                other => bail!("Prototype C GPU returned invalid exit encoding {other:?}"),
            };
            Ok(SparseOutcome {
                candidate_id: event.candidate_id,
                scenario_id: event.scenario_id,
                hit,
            })
        })
        .collect::<Result<Vec<_>>>()
        .context("stitch Prototype C GPU outcomes")
}

#[cfg(all(test, feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
mod tests {
    use super::*;
    use crate::gpu_native::prototype_bc::{FirstHitRequest, prototype_c_event_first_hit};

    #[test]
    fn gpu_event_first_hit_matches_reference_when_adapter_is_available() {
        let highs = [100.0, 101.0, 106.0, 103.0, 104.0, 105.0];
        let lows = [100.0, 99.0, 98.0, 94.0, 96.0, 97.0];
        let path = PricePath {
            highs: &highs,
            lows: &lows,
        };
        let events = [EntryEvent {
            candidate_id: 11,
            scenario_id: 101,
            request: FirstHitRequest {
                entry_bar: 0,
                last_bar: 5,
                direction: PositionDirection::Long,
                stop_price: 95.0,
                target_price: 105.0,
                same_bar_precedence: SameBarPrecedence::StopFirst,
            },
        }];
        let expected = prototype_c_event_first_hit(path, &events).unwrap();
        let actual = match try_prototype_c_gpu_first_hit(path, &events) {
            Ok(actual) => actual,
            Err(error) => {
                eprintln!("Prototype C direct GPU test skipped: {error:#}");
                return;
            }
        };
        assert_eq!(actual, expected);
    }
}
