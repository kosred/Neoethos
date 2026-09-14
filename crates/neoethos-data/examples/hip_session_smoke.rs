//! A bounded, real-device Data Session-v2 smoke/parity entry point.
//!
//! Run the `hip_session_smoke` example with `gpu-hip-session` and `--device N`.
//! A missing/incorrect HIP device or artifact is an error, never a skip or CPU
//! fallback. The CPU comparison is the EXISTING production Session oracle, not
//! an independently proved mathematical reference. This fixture is neither a
//! trading backtest, OOS/CPCV evidence, nor whole-bot/whole-feature-store parity.

use anyhow::{Context as _, Result, ensure};
use neoethos_data::Ohlcv;
#[path = "support/hip_producer_smoke_v1.rs"]
mod support;
use neoethos_data::core::gpu_hip_session_v1::{HIP_SESSION_COLUMN_NAMES_V1, PreparedHipSessionV1};
use neoethos_data::core::session_features::compute_session_feature_columns_f64;
use neoethos_gpu_cuda::hip_runtime_v1::HipRunLeaseV1;
use serde_json::json;
use sha2::{Digest, Sha256};
use support::{compare_terminal, hex};

const SCHEMA: &str = "neoethos.data.hip-session-smoke.v1";
const FIXTURE_ROWS: usize = 31;

fn fixture(flat_zero_volume: bool) -> Ohlcv {
    // Exact UTC timestamps, including both sides of all Session-v2 opening /
    // closing boundaries, two day resets and deliberate gaps. No missing input
    // rows are invented between those explicitly supplied timestamps.
    const DAY_MINUTES: [i64; 15] = [
        0, 1, 419, 420, 421, 479, 480, 719, 720, 721, 959, 960, 1259, 1260, 1439,
    ];
    const JAN_1_2024_UTC_MS: i64 = 1_704_067_200_000;
    let timestamp: Vec<_> = (0..2)
        .flat_map(|day| {
            DAY_MINUTES
                .into_iter()
                .map(move |minute| day * 1440 + minute)
        })
        .chain([2880])
        .map(|minute| JAN_1_2024_UTC_MS + minute * 60_000)
        .collect();
    let mut input = Ohlcv {
        timestamp: Some(timestamp),
        open: Vec::new(),
        high: Vec::new(),
        low: Vec::new(),
        close: Vec::new(),
        volume: Some(Vec::new()),
    };
    for row in 0..FIXTURE_ROWS {
        let open = if flat_zero_volume {
            2.0
        } else {
            2.0 + (row % 7) as f64 / 16.0
        };
        let close = if flat_zero_volume {
            open
        } else {
            open + if row % 2 == 0 { 0.125 } else { -0.125 }
        };
        input.open.push(open);
        input.high.push(if flat_zero_volume {
            open
        } else {
            open.max(close) + 0.25
        });
        input.low.push(if flat_zero_volume {
            open
        } else {
            open.min(close) - 0.25
        });
        input.close.push(close);
        input
            .volume
            .as_mut()
            .unwrap()
            .push(if flat_zero_volume || row % 5 == 0 {
                0.0
            } else {
                1.0 + (row % 3) as f64
            });
    }
    input
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 3 && args[1] == "--device",
        "usage: hip_session_smoke --device <HIP ordinal>"
    );
    let ordinal: u32 = args[2]
        .parse()
        .context("HIP ordinal must be a nonnegative u32")?;
    let lease = HipRunLeaseV1::acquire(ordinal)
        .context("a real HIP device is required; no skip or fallback")?;
    println!(
        "{}",
        json!({
            "type": "metadata", "schema": SCHEMA, "backend": "amd-hip",
            "scope": "two-fixture-session-v2-smoke-only", "cpu_oracle": "existing-production-Session-v2",
            "independent_mathematical_proof": false, "trading_or_oos_proof": false,
            "device_ordinal": lease.identity().device_ordinal(), "device_uuid": hex(&lease.identity().device_uuid()),
            "architecture": lease.identity().architecture(), "runtime_version": lease.identity().runtime_version(),
            "driver_version": lease.identity().driver_version(), "lease_id": lease.identity().lease_id(),
            "stream_id": lease.identity().stream_id(),
        })
    );
    let mut mismatches = 0usize;
    for (id, flat) in [
        ("gapped-session-day-boundaries", false),
        ("flat-zero-volume", true),
    ] {
        let input = fixture(flat);
        let prepared = PreparedHipSessionV1::preflight(&input)?;
        let cpu = compute_session_feature_columns_f64(&input)?;
        let session = prepared.materialize(&lease)?;
        // THE ONLY FEATURE READBACK BOUNDARY: the bounded producer has finished.
        // No original/derived feature matrix is returned during materialization.
        let diagnostic = session.read_terminal_diagnostic(input.len() * 23 * 9)?;
        let values = diagnostic.values;
        let validity = diagnostic.validity;
        let comparison = compare_terminal(
            &cpu,
            &HIP_SESSION_COLUMN_NAMES_V1,
            input.len(),
            &values,
            &validity,
        )?;
        mismatches += comparison.value_mismatches + comparison.validity_mismatches;
        println!(
            "{}",
            json!({
                "type": "fixture", "schema": SCHEMA, "id": id,
                "rows": input.len(), "columns": cpu.len(), "cells": input.len() * cpu.len(),
                "input_sha256": hex(&prepared.input_sha256()), "artifact_sha256": hex(&session.identity().artifact_sha256()),
                "build_manifest_sha256": hex(&session.identity().manifest_sha256()), "target": session.identity().target(),
                "value_bit_mismatches": comparison.value_mismatches, "validity_mismatches": comparison.validity_mismatches,
                "cpu_values_sha256": comparison.cpu_value_sha256, "hip_values_sha256": hex(&Sha256::digest(&values)),
                "cpu_validity_sha256": comparison.cpu_validity_sha256, "hip_validity_sha256": hex(&Sha256::digest(&validity)),
                "first_mismatch": comparison.first_mismatch, "terminal_d2h_bytes": values.len() + validity.len(),
            })
        );
        session.try_close()?;
    }
    // Error owns a thread-confined lease, so deliberately format its diagnostic
    // instead of trying to convert that !Send/!Sync owner into anyhow::Error.
    lease
        .try_close()
        .map_err(|error| anyhow::anyhow!("HIP lease cleanup failed: {error}"))?;
    println!(
        "{}",
        json!({
            "type": "summary", "schema": SCHEMA, "fixtures": 2, "cells": FIXTURE_ROWS * 23 * 2,
            "mismatches": mismatches, "device_executed": true, "cleanup_completed": true,
            "passed": mismatches == 0, "whole_pipeline_parity": false,
        })
    );
    ensure!(
        mismatches == 0,
        "HIP Session smoke found {mismatches} exact mismatches"
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!(
                "{}",
                json!({"type": "error", "schema": SCHEMA,
                "message": format!("{error:#}"), "passed": false, "gpu_fallback": false})
            );
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_and_comparator_host_controls_are_not_device_evidence() {
        for flat in [false, true] {
            let input = fixture(flat);
            assert_eq!(input.len(), FIXTURE_ROWS);
            PreparedHipSessionV1::preflight(&input).unwrap();
            let cpu = compute_session_feature_columns_f64(&input).unwrap();
            let mut values: Vec<_> = cpu
                .iter()
                .flat_map(|column| {
                    column
                        .values
                        .iter()
                        .flat_map(|value| value.to_bits().to_le_bytes())
                })
                .collect();
            let mut validity: Vec<_> = cpu
                .iter()
                .flat_map(|column| column.validity.iter().map(|value| value.code()))
                .collect();
            let control = compare_terminal(
                &cpu,
                &HIP_SESSION_COLUMN_NAMES_V1,
                input.len(),
                &values,
                &validity,
            )
            .unwrap();
            assert_eq!(
                (control.value_mismatches, control.validity_mismatches),
                (0, 0)
            );
            values[0] ^= 1;
            let changed = compare_terminal(
                &cpu,
                &HIP_SESSION_COLUMN_NAMES_V1,
                input.len(),
                &values,
                &validity,
            )
            .unwrap();
            assert_eq!(
                (changed.value_mismatches, changed.validity_mismatches),
                (1, 0)
            );
            values[0] ^= 1;
            validity[0] ^= 1;
            let changed = compare_terminal(
                &cpu,
                &HIP_SESSION_COLUMN_NAMES_V1,
                input.len(),
                &values,
                &validity,
            )
            .unwrap();
            assert_eq!(
                (changed.value_mismatches, changed.validity_mismatches),
                (0, 1)
            );
            assert!(
                compare_terminal(
                    &cpu,
                    &HIP_SESSION_COLUMN_NAMES_V1,
                    input.len(),
                    &values[..values.len() - 1],
                    &validity
                )
                .is_err()
            );
        }
    }
}
