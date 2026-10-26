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

pub fn run(args: &[String]) -> Result<()> {
    let research = flag(args, "--research")
        .context("export-live-portfolio requires --research <research.json>")?;
    let out = flag(args, "--out")
        .context("export-live-portfolio requires --out <path.live_portfolio.json>")?;
    ensure_requested_only(args, &["--research", "--out"])?;

    let bytes = std::fs::read(&research)
        .with_context(|| format!("research artifact {research} is not readable"))?;
    let wrapper: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("research artifact {research} is not valid JSON"))?;
    let discovery = wrapper
        .get("discovery_result")
        .context("research artifact has no discovery_result")?;
    let result: neoethos_search::DiscoveryResult = serde_json::from_value(discovery.clone())
        .context("research artifact discovery_result does not match the DiscoveryResult schema")?;
    if result.portfolio.is_empty() {
        bail!("research artifact has an empty portfolio; nothing to export");
    }
    neoethos_search::save_live_portfolio_json(&out, &result)
        .with_context(|| format!("failed to write live portfolio {out}"))?;
    println!(
        "exported live portfolio: genes={} symbol={} base={} out={out}",
        result.portfolio.len(),
        result.symbols.first().map(String::as_str).unwrap_or("?"),
        result.base_timeframe
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