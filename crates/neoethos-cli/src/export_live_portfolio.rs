//! export-live-portfolio --research <research.json> --out <path> - convert a
//! completed canonical-research artifact into the self-describing
//! live_portfolio.json artifact that forward-test consumes.
//!
//! This is a pure, non-financial conversion: it reads the discovery_result
//! written by canonical-research and re-emits it through
//! neoethos_search::save_live_portfolio_json, which binds the completed search
//! receipt (normalization state, effective feature names, genes, symbol and
//! timeframes) into the shared v3 envelope. No trading capability is required
//! and no orders are produced.
//!
//! Why it exists (neo-671, 2026-10-26): the faithful OOS path
//! (forward-test --portfolio ...) needs a live portfolio artifact, while
//! canonical-research is deliberately research-only. Without this bridge the
//! searched candidates could not be validated out of sample at all.

use anyhow::{bail, Context, Result};

#[derive(serde::Deserialize)]
struct ResearchWrapper {
    discovery_result: neoethos_search::DiscoveryResult,
}

pub fn run(args: &[String]) -> Result<()> {
    let research = flag(args, "--research")
        .context("export-live-portfolio requires --research <research.json>")?;
    let out = flag(args, "--out")
        .context("export-live-portfolio requires --out <path.live_portfolio.json>")?;
    ensure_requested_only(args, &["--research", "--out"])?;

    let bytes = std::fs::read(&research)
        .with_context(|| format!("research artifact {research} is not readable"))?;
    // Deserialize straight into the wrapper instead of a generic Value: the
    // canonical artifact can exceed 500 MB, and a Value clone would double the
    // peak memory for no benefit.
    let wrapper: ResearchWrapper = serde_json::from_slice(&bytes)
        .with_context(|| format!("research artifact {research} is not valid JSON"))?;
    let result = wrapper.discovery_result;
    if result.portfolio.is_empty() {
        bail!("research artifact has an empty portfolio; nothing to export");
    }
    neoethos_search::save_live_portfolio_json(&out, &result)
        .with_context(|| format!("failed to write live portfolio {out}"))?;
    println!(
        "exported live portfolio: genes={} out={out}",
        result.portfolio.len()
    );
    Ok(())
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn ensure_requested_only(args: &[String], allowed: &[&str]) -> Result<()> {
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if !arg.starts_with("--") {
            bail!("unexpected positional argument {arg}");
        }
        if !allowed.contains(&arg.as_str()) {
            bail!("unsupported flag {arg}");
        }
        index += 2;
    }
    Ok(())
}