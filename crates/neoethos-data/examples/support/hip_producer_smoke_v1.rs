//! Shared exact terminal comparison for bounded HIP producer smoke examples.
use anyhow::{Context as _, Result, ensure};
use neoethos_data::core::features::FeatureColumnF64;
use serde_json::json;
use sha2::{Digest, Sha256};

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) struct Comparison {
    pub value_mismatches: usize,
    pub validity_mismatches: usize,
    pub cpu_value_sha256: String,
    pub cpu_validity_sha256: String,
    pub first_mismatch: Option<serde_json::Value>,
}

pub(crate) fn compare_terminal(
    cpu: &[FeatureColumnF64],
    expected_names: &[&str],
    rows: usize,
    values: &[u8],
    validity: &[u8],
) -> Result<Comparison> {
    ensure!(
        cpu.len() == expected_names.len(),
        "CPU producer column count drift"
    );
    let cells = rows
        .checked_mul(cpu.len())
        .context("comparison cell count overflow")?;
    ensure!(
        values.len()
            == cells
                .checked_mul(8)
                .context("comparison byte count overflow")?
            && validity.len() == cells,
        "terminal HIP output extent differs from exact CPU shape"
    );
    let mut result = Comparison {
        value_mismatches: 0,
        validity_mismatches: 0,
        cpu_value_sha256: String::new(),
        cpu_validity_sha256: String::new(),
        first_mismatch: None,
    };
    let mut value_hash = Sha256::new();
    let mut validity_hash = Sha256::new();
    for (column_index, column) in cpu.iter().enumerate() {
        ensure!(
            column.name == expected_names[column_index]
                && column.values.len() == rows
                && column.validity.len() == rows,
            "CPU producer column identity or extent drift at column {column_index}"
        );
        for row in 0..rows {
            let cell = column_index * rows + row;
            let cpu_bits = column.values[row].to_bits();
            let hip_bits = u64::from_le_bytes(values[cell * 8..cell * 8 + 8].try_into()?);
            let cpu_validity = column.validity[row].code();
            let hip_validity = validity[cell];
            value_hash.update(cpu_bits.to_le_bytes());
            validity_hash.update([cpu_validity]);
            // All cells, including canonical invalid NaNs and signed zero, are
            // compared exactly. No tolerance, filtering or NaN-equal shortcut.
            result.value_mismatches += usize::from(cpu_bits != hip_bits);
            result.validity_mismatches += usize::from(cpu_validity != hip_validity);
            if result.first_mismatch.is_none()
                && (cpu_bits != hip_bits || cpu_validity != hip_validity)
            {
                result.first_mismatch = Some(json!({
                    "column": column.name, "row": row,
                    "cpu_bits": format!("{cpu_bits:016x}"), "hip_bits": format!("{hip_bits:016x}"),
                    "cpu_validity": cpu_validity, "hip_validity": hip_validity,
                }));
            }
        }
    }
    result.cpu_value_sha256 = hex(&value_hash.finalize());
    result.cpu_validity_sha256 = hex(&validity_hash.finalize());
    Ok(result)
}
