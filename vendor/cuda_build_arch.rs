//! Shared validation for CUDA architectures passed to vendored model builds.

use std::collections::HashSet;

pub(crate) const ENV_NAME: &str = "NEOETHOS_CUDA_ARCHS";

/// Read and validate the explicit project architecture set.
///
/// The input is a semicolon-separated set of numeric compute capabilities such
/// as `86;89`. Magic values such as `native` are rejected because they make
/// otherwise identical builds depend on the build host.
pub(crate) fn required_cuda_arch_numbers() -> Vec<u32> {
    let raw = std::env::var(ENV_NAME).unwrap_or_else(|_| {
        panic!("CUDA builds require {ENV_NAME}; use an explicit list such as `89` or `86;89`")
    });
    parse_cuda_architectures(&raw)
        .unwrap_or_else(|error| panic!("invalid {ENV_NAME}={raw:?}: {error}"))
}

#[cfg(test)]
fn normalize_cuda_architectures(raw: &str) -> Result<String, String> {
    Ok(parse_cuda_architectures(raw)?
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(";"))
}

fn parse_cuda_architectures(raw: &str) -> Result<Vec<u32>, String> {
    let mut normalized = Vec::new();
    let mut seen = HashSet::new();

    for item in raw.split(';') {
        let item = item.trim();
        if item.is_empty() {
            return Err("expected a semicolon-separated list such as `86;89`".into());
        }

        if !item.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "unsupported architecture {item:?}; use numeric compute capabilities only"
            ));
        }
        let value = item
            .parse::<u32>()
            .map_err(|_| format!("architecture {item:?} is outside the supported numeric range"))?;
        if value == 0 {
            return Err("architecture zero is invalid".into());
        }
        if !seen.insert(value) {
            return Err(format!("duplicate architecture {item:?}"));
        }
        normalized.push(value);
    }

    if normalized.is_empty() {
        return Err("the architecture list cannot be empty".into());
    }
    normalized.sort_unstable();
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::normalize_cuda_architectures;

    #[test]
    fn accepts_numeric_cmake_architecture_lists() {
        assert_eq!(normalize_cuda_architectures("89").unwrap(), "89");
        assert_eq!(normalize_cuda_architectures("89; 86").unwrap(), "86;89");
    }

    #[test]
    fn rejects_host_dependent_or_nvcc_gencode_syntax() {
        for invalid in [
            "",
            "native",
            "all",
            "89-real",
            "compute_89,code=sm_89",
            "89;89",
        ] {
            assert!(
                normalize_cuda_architectures(invalid).is_err(),
                "{invalid:?} unexpectedly passed"
            );
        }
    }
}
