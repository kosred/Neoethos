//! Standalone CPU side of the production exact-log backend parity probe.
//! Build with rustc --edition=2021 -O; run with the explicit shared CSV path.
//! The normal Cargo test also checks the fixture without requiring any GPU.
#[path = "../../neoethos-data/src/core/quant_exact_math_v3.rs"]
mod production;

use std::{collections::HashSet, fmt::Write, path::Path};

const SENTINEL: u64 = 0x0123_4567_89ab_cdef;
const SCHEMA: &str = "neoethos.exact-log-backend-parity.v1";

fn bits(text: &str) -> Result<u64, String> {
    if text.len() != 16 || !text.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("invalid 16-digit bits: {text}"));
    }
    u64::from_str_radix(text, 16).map_err(|e| e.to_string())
}

fn ordered(bits: u64) -> u64 {
    if bits >> 63 == 0 {
        bits | (1 << 63)
    } else {
        !bits
    }
}

fn run(path: &Path) -> Result<(String, usize, usize), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut output = String::new();
    let mut seen = HashSet::new();
    let mut count = 0;
    let mut failures = 0;
    for line in text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let f: Vec<_> = line.split(',').collect();
        if f.len() != 7
            || f[0].is_empty()
            || !f[0].bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
            || !seen.insert(f[0])
        {
            return Err(format!("malformed/duplicate vector: {line}"));
        }
        count += 1;
        if count > 4096 {
            return Err("fixture exceeds 4096 cases".into());
        }
        let a_bits = bits(f[2])?;
        let b_bits = bits(f[3])?;
        let c_bits = bits(f[4])?;
        let (a, b, c) = (
            f64::from_bits(a_bits),
            f64::from_bits(b_bits),
            f64::from_bits(c_bits),
        );
        let value = match f[1] {
            "log" => production::quant_log_positive_f64_v3(a),
            "add" => Some(a + b),
            "sub" => Some(a - b),
            "mul" => Some(a * b),
            "div" => Some(a / b),
            "unfused" => {
                let product = a * b;
                Some(product + c)
            }
            "fused" => Some(a.mul_add(b, c)),
            op => return Err(format!("unknown operation: {op}")),
        };
        let accepted = value.is_some();
        let output_bits = value.map(f64::to_bits).unwrap_or(SENTINEL);
        let expected_accepted = f[5] != "reject";
        if !expected_accepted && (f[1] != "log" || f[6] != "-") {
            return Err("only log-domain rejection is supported".into());
        }
        if f[5] == "-" && f[6] == "-" {
            return Err("missing independent expectation".into());
        }
        let mut passed = accepted == expected_accepted;
        if f[5] == "reject" {
            passed &= output_bits == SENTINEL;
        } else if f[5] != "-" {
            passed &= output_bits == bits(f[5])?;
        }
        let ulp = if f[6] == "-" {
            None
        } else {
            if f[1] != "log" {
                return Err("accuracy reference is only for log".into());
            }
            let reference = bits(f[6])?;
            if !f64::from_bits(reference).is_finite() {
                return Err("nonfinite accuracy reference".into());
            }
            let distance = ordered(output_bits).abs_diff(ordered(reference));
            passed &= accepted && value.is_some_and(f64::is_finite) && distance <= 1;
            Some(distance)
        };
        if !passed {
            failures += 1;
        }
        writeln!(output,
            "{{\"type\":\"case\",\"id\":{:?},\"operation\":{:?},\"input_bits\":\"{a_bits:016x}\",\"b_bits\":\"{b_bits:016x}\",\"c_bits\":\"{c_bits:016x}\",\"accepted\":{accepted},\"output_bits\":\"{output_bits:016x}\",\"accuracy_ulp\":{},\"passed\":{passed}}}",
            f[0], f[1], ulp.map_or_else(|| "null".into(), |u| u.to_string())
        ).map_err(|e| e.to_string())?;
    }
    if count == 0 {
        return Err("empty fixture".into());
    }
    Ok((output, count, failures))
}

#[cfg(not(test))]
fn main() {
    println!(
        "{{\"type\":\"metadata\",\"schema\":{:?},\"backend\":\"cpu\",\"role\":\"production_cpu\",\"architecture\":{:?},\"operation_schedule\":{:?},\"authority_commit\":{:?},\"authority_sha256\":{:?},\"authority_source\":{:?},\"authority_receipt\":{:?}}}",
        SCHEMA,
        std::env::consts::ARCH,
        production::QUANT_LOG_OPERATION_SCHEDULE_V3,
        production::QUANT_OPENLIBM_COMMIT_V3,
        production::QUANT_OPENLIBM_E_LOG_SOURCE_SHA256_V3,
        production::QUANT_OPENLIBM_E_LOG_SOURCE_V3,
        production::QUANT_OPENLIBM_E_LOG_RECEIPT_V3
    );
    let args: Vec<_> = std::env::args_os().collect();
    let result = if args.len() == 2 {
        run(Path::new(&args[1]))
    } else {
        Err("usage: exact_log_cpu_parity_v1 <fixture.csv>".into())
    };
    match result {
        Ok((rows, cases, failures)) => {
            print!("{rows}");
            println!(
                "{{\"type\":\"summary\",\"cases\":{cases},\"failures\":{failures},\"device_executed\":false}}"
            );
            if failures != 0 {
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("{error}");
            println!(
                "{{\"type\":\"summary\",\"cases\":0,\"failures\":1,\"device_executed\":false}}"
            );
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
#[test]
fn production_cpu_matches_shared_backend_parity_vectors() {
    assert_eq!(SCHEMA, "neoethos.exact-log-backend-parity.v1");
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/exact_log_backend_vectors_v1.csv");
    let (rows, cases, failures) = run(&path).expect("read exact-log parity vectors");
    assert_eq!(cases, 48, "intentional bounded fixture coverage");
    assert_eq!(failures, 0, "{rows}");
}
