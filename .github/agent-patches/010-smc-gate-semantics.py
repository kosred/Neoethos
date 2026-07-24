from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text, encoding="utf-8")


def replace_exact(text: str, old: str, new: str, *, label: str, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"{label}: expected {count} exact matches, found {actual}")
    return text.replace(old, new, count)


def find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    state = "code"
    i = open_index
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == '"':
                state = "string"
            elif ch == "'":
                # Rust lifetimes are common. Treat as a char only when a closing
                # quote is immediately plausible; otherwise leave it as code.
                if i + 2 < len(text) and text[i + 2] == "'":
                    state = "char"
            elif ch == "/" and nxt == "/":
                state = "line_comment"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block_comment"
                i += 1
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "string":
            if ch == "\\":
                i += 1
            elif ch == '"':
                state = "code"
        elif state == "char":
            if ch == "\\":
                i += 1
            elif ch == "'":
                state = "code"
        elif state == "line_comment":
            if ch == "\n":
                state = "code"
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        i += 1
    raise RuntimeError(f"unmatched brace at byte {open_index}")


def add_field_to_struct_literals(text: str, type_name: str, field: str, value: str) -> tuple[str, int]:
    starts: list[tuple[int, int]] = []
    pattern = re.compile(rf"\b{re.escape(type_name)}\s*\{{")
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 32) : match.start()]
        if re.search(r"(?:pub\s+)?struct\s+$", prefix):
            continue
        open_index = text.find("{", match.start(), match.end())
        close_index = find_matching_brace(text, open_index)
        body = text[open_index + 1 : close_index]
        if re.search(rf"\b{re.escape(field)}\s*:", body):
            continue
        starts.append((open_index, close_index))

    for open_index, close_index in reversed(starts):
        line_start = text.rfind("\n", 0, close_index) + 1
        closing_indent = text[line_start:close_index]
        if closing_indent.strip():
            closing_indent = ""
        body = text[open_index + 1 : close_index]
        stripped = body.rstrip()
        whitespace = body[len(stripped) :]
        if stripped and not stripped.endswith(","):
            stripped += ","
        field_indent = closing_indent + "    "
        new_body = f"{stripped}{whitespace}\n{field_indent}{field}: {value},\n{closing_indent}"
        text = text[: open_index + 1] + new_body + text[close_index:]
    return text, len(starts)


# ── SearchResult carries the exact gate used by the final GA evaluation. ─────
strategy_path = "crates/neoethos-search/src/genetic/strategy_gene.rs"
strategy = read(strategy_path)
strategy = replace_exact(
    strategy,
    "pub struct SearchResult {\n    pub genes: Vec<Gene>,\n    pub metrics: Vec<[f64; 11]>,\n}",
    "pub struct SearchResult {\n    pub genes: Vec<Gene>,\n    pub metrics: Vec<[f64; 11]>,\n    /// Effective SMC gate used for the metrics in this result. This is the\n    /// annealed final-generation value, not the static runtime start value.\n    pub effective_smc_gate_threshold: f32,\n}",
    label="SearchResult field",
)
write(strategy_path, strategy)

search_path = "crates/neoethos-search/src/genetic/search_engine.rs"
search = read(search_path)
search = replace_exact(
    search,
    "    let metrics = evaluate_genes(features, ohlcv, &genes, &EvaluationConfig::default())?;\n",
    "    let eval_cfg = EvaluationConfig::default();\n    let metrics = evaluate_genes(features, ohlcv, &genes, &eval_cfg)?;\n",
    label="random_search evaluation config",
)
search, result_literals = add_field_to_struct_literals(
    search,
    "SearchResult",
    "effective_smc_gate_threshold",
    "eval_cfg.smc_gate_threshold",
)
if result_literals < 7:
    raise RuntimeError(f"expected at least 7 SearchResult constructors, found {result_literals}")
write(search_path, search)

# Checkpoints preserve the resolved gate and deliberately bump schema.
checkpoint_path = "crates/neoethos-search/src/checkpoint.rs"
checkpoint = read(checkpoint_path)
checkpoint = replace_exact(
    checkpoint,
    "const CHECKPOINT_SCHEMA_VERSION: u32 = 2;",
    "const CHECKPOINT_SCHEMA_VERSION: u32 = 3;",
    label="checkpoint schema version",
)
checkpoint = replace_exact(
    checkpoint,
    "    pub genes: Vec<Gene>,\n    pub metrics: Vec<[f64; 11]>,\n}",
    "    pub genes: Vec<Gene>,\n    pub metrics: Vec<[f64; 11]>,\n    pub effective_smc_gate_threshold: f32,\n}",
    label="checkpoint gate field",
)
checkpoint = replace_exact(
    checkpoint,
    "            genes: result.genes,\n            metrics: result.metrics,\n",
    "            genes: result.genes,\n            metrics: result.metrics,\n            effective_smc_gate_threshold: result.effective_smc_gate_threshold,\n",
    label="checkpoint constructor gate",
)
checkpoint, checkpoint_literals = add_field_to_struct_literals(
    checkpoint,
    "SearchResult",
    "effective_smc_gate_threshold",
    "self.effective_smc_gate_threshold",
)
if checkpoint_literals != 1:
    raise RuntimeError(f"expected one checkpoint SearchResult literal, found {checkpoint_literals}")
write(checkpoint_path, checkpoint)

# ── Discovery carries the gate through IS, WF, CPCV and held-out replay. ─────
discovery_path = "crates/neoethos-search/src/discovery.rs"
discovery = read(discovery_path)

discovery = replace_exact(
    discovery,
    "        cfg.growth_objective = matches!(self.mode, DiscoveryMode::Risky);\n        cfg\n    }\n}",
    "        cfg.growth_objective = matches!(self.mode, DiscoveryMode::Risky);\n        cfg\n    }\n\n    pub fn evaluation_config_with_smc_gate(\n        &self,\n        price_hint: Option<f64>,\n        effective_smc_gate_threshold: f32,\n    ) -> EvaluationConfig {\n        let mut cfg = self.evaluation_config(price_hint);\n        cfg.smc_gate_threshold = effective_smc_gate_threshold;\n        cfg\n    }\n}",
    label="resolved evaluation config method",
)

discovery = replace_exact(
    discovery,
    "    pub effective_feature_names: Vec<String>,\n    pub validation_gates: DiscoveryValidationGates,\n",
    "    pub effective_feature_names: Vec<String>,\n    /// Final annealed SMC gate used by the GA and every post-search replay.\n    pub effective_smc_gate_threshold: f32,\n    pub validation_gates: DiscoveryValidationGates,\n",
    label="DiscoveryResult gate field",
)

# Every synthetic/test literal gets an explicit non-production sentinel. The
# real finalize constructor is replaced with the actual gate below.
changed_discovery_literals = 0
for path in ROOT.rglob("*.rs"):
    text = path.read_text(encoding="utf-8")
    updated, changed = add_field_to_struct_literals(
        text,
        "DiscoveryResult",
        "effective_smc_gate_threshold",
        "f32::NAN",
    )
    if changed:
        path.write_text(updated, encoding="utf-8")
        changed_discovery_literals += changed
if changed_discovery_literals < 1:
    raise RuntimeError("no DiscoveryResult literals were updated")
discovery = read(discovery_path)

# Search result value is captured before moving its genes into finalize.
discovery = replace_exact(
    discovery,
    "    let stage1_count = search.genes.len();\n",
    "    let effective_smc_gate_threshold = search.effective_smc_gate_threshold;\n    let stage1_count = search.genes.len();\n",
    label="capture final SMC gate",
)
discovery = replace_exact(
    discovery,
    "        config,\n        effective_feature_names,\n        &mut funnel,\n",
    "        config,\n        effective_smc_gate_threshold,\n        effective_feature_names,\n        &mut funnel,\n",
    label="finalize call gate",
)
discovery = replace_exact(
    discovery,
    "    config: &DiscoveryConfig,\n    effective_feature_names: Vec<String>,\n    funnel: &mut crate::funnel_profile::FunnelProfile,\n",
    "    config: &DiscoveryConfig,\n    effective_smc_gate_threshold: f32,\n    effective_feature_names: Vec<String>,\n    funnel: &mut crate::funnel_profile::FunnelProfile,\n",
    label="finalize signature gate",
)

# Internal validation functions receive the same explicit gate.
discovery = replace_exact(
    discovery,
    "    config: &DiscoveryConfig,\n    months: &[i64],\n    days: &[i64],\n    pbo_candidates: &[Gene],\n) -> Result<(bool, usize, f64, Option<f64>, bool)> {",
    "    config: &DiscoveryConfig,\n    effective_smc_gate_threshold: f32,\n    months: &[i64],\n    days: &[i64],\n    pbo_candidates: &[Gene],\n) -> Result<(bool, usize, f64, Option<f64>, bool)> {",
    label="CPCV signature gate",
)
discovery = replace_exact(
    discovery,
    "    config: &DiscoveryConfig,\n    pbo_candidates: &[Gene],\n    trials_tested: usize,\n) -> Result<(\n",
    "    config: &DiscoveryConfig,\n    effective_smc_gate_threshold: f32,\n    pbo_candidates: &[Gene],\n    trials_tested: usize,\n) -> Result<(\n",
    label="validation artifact signature gate",
)

discovery = replace_exact(
    discovery,
    "    let eval_config = config.evaluation_config(ohlcv.close.last().copied());\n",
    "    let eval_config = config.evaluation_config_with_smc_gate(\n        ohlcv.close.last().copied(),\n        effective_smc_gate_threshold,\n    );\n",
    label="CPCV resolved gate",
)
discovery = replace_exact(
    discovery,
    "    let wf_eval_config = config.evaluation_config(ohlcv.close.last().copied());\n",
    "    let wf_eval_config = config.evaluation_config_with_smc_gate(\n        ohlcv.close.last().copied(),\n        effective_smc_gate_threshold,\n    );\n",
    label="walk-forward resolved gate",
)
discovery = replace_exact(
    discovery,
    "            config,\n            &months,\n            &days,\n            pbo_candidates,\n",
    "            config,\n            effective_smc_gate_threshold,\n            &months,\n            &days,\n            pbo_candidates,\n",
    label="CPCV call gate",
)
discovery = replace_exact(
    discovery,
    "                config,\n                &pbo_candidates,\n                ranked_total,\n",
    "                config,\n                effective_smc_gate_threshold,\n                &pbo_candidates,\n                ranked_total,\n",
    label="validation artifact call gate",
)

discovery = replace_exact(
    discovery,
    "    let eval_config_for_signals = config.evaluation_config(ohlcv.close.last().copied());\n",
    "    let eval_config_for_signals = config.evaluation_config_with_smc_gate(\n        ohlcv.close.last().copied(),\n        effective_smc_gate_threshold,\n    );\n",
    label="post-search signal gate",
)
discovery = replace_exact(
    discovery,
    "        let eval_cfg_rb = config.evaluation_config(ohlcv.close.last().copied());\n",
    "        let eval_cfg_rb = config.evaluation_config_with_smc_gate(\n            ohlcv.close.last().copied(),\n            effective_smc_gate_threshold,\n        );\n",
    label="robustness replay gate",
)

# Preserve public APIs with compatibility wrappers, while the holdout driver
# calls the explicit resolved-gate variants.
forward_signature = "pub fn compute_discovery_forward_test_artifacts(\n    portfolio: &[Gene],\n    effective_feature_names: &[String],\n    tail_features: &FeatureFrame,\n    tail_ohlcv: &Ohlcv,\n    config: &DiscoveryConfig,\n) -> Result<Vec<ForwardTestValidationArtifactFile>> {"
forward_replacement = "pub fn compute_discovery_forward_test_artifacts(\n    portfolio: &[Gene],\n    effective_feature_names: &[String],\n    tail_features: &FeatureFrame,\n    tail_ohlcv: &Ohlcv,\n    config: &DiscoveryConfig,\n) -> Result<Vec<ForwardTestValidationArtifactFile>> {\n    let effective_smc_gate_threshold = config\n        .evaluation_config(tail_ohlcv.close.last().copied())\n        .smc_gate_threshold;\n    compute_discovery_forward_test_artifacts_with_smc_gate(\n        portfolio,\n        effective_feature_names,\n        tail_features,\n        tail_ohlcv,\n        config,\n        effective_smc_gate_threshold,\n    )\n}\n\npub fn compute_discovery_forward_test_artifacts_with_smc_gate(\n    portfolio: &[Gene],\n    effective_feature_names: &[String],\n    tail_features: &FeatureFrame,\n    tail_ohlcv: &Ohlcv,\n    config: &DiscoveryConfig,\n    effective_smc_gate_threshold: f32,\n) -> Result<Vec<ForwardTestValidationArtifactFile>> {"
discovery = replace_exact(
    discovery,
    forward_signature,
    forward_replacement,
    label="forward explicit gate API",
)

# Remove the now-obsolete diagnostic-only F-315 block.
pattern = re.compile(
    r"\n    // \*\*F-315 \(2026-05-29\).*?\n    // Each portfolio gene's forward-test replay",
    re.DOTALL,
)
discovery, removed = pattern.subn(
    "\n    // Each portfolio gene's forward-test replay",
    discovery,
    count=1,
)
if removed != 1:
    raise RuntimeError(f"F-315 diagnostic block: expected one match, found {removed}")

discovery = replace_exact(
    discovery,
    "            let evaluation_config = config.evaluation_config(tail_ohlcv.close.last().copied());\n",
    "            let evaluation_config = config.evaluation_config_with_smc_gate(\n                tail_ohlcv.close.last().copied(),\n                effective_smc_gate_threshold,\n            );\n",
    label="forward replay resolved gate",
    count=2,
)

prop_signature = "pub fn compute_discovery_prop_firm_artifacts(\n    portfolio: &[Gene],\n    effective_feature_names: &[String],\n    tail_features: &FeatureFrame,\n    tail_ohlcv: &Ohlcv,\n    config: &DiscoveryConfig,\n    rules: PropFirmRiskRules,\n) -> Result<Vec<PropFirmRiskValidationArtifactFile>> {"
prop_replacement = "pub fn compute_discovery_prop_firm_artifacts(\n    portfolio: &[Gene],\n    effective_feature_names: &[String],\n    tail_features: &FeatureFrame,\n    tail_ohlcv: &Ohlcv,\n    config: &DiscoveryConfig,\n    rules: PropFirmRiskRules,\n) -> Result<Vec<PropFirmRiskValidationArtifactFile>> {\n    let effective_smc_gate_threshold = config\n        .evaluation_config(tail_ohlcv.close.last().copied())\n        .smc_gate_threshold;\n    compute_discovery_prop_firm_artifacts_with_smc_gate(\n        portfolio,\n        effective_feature_names,\n        tail_features,\n        tail_ohlcv,\n        config,\n        effective_smc_gate_threshold,\n        rules,\n    )\n}\n\npub fn compute_discovery_prop_firm_artifacts_with_smc_gate(\n    portfolio: &[Gene],\n    effective_feature_names: &[String],\n    tail_features: &FeatureFrame,\n    tail_ohlcv: &Ohlcv,\n    config: &DiscoveryConfig,\n    effective_smc_gate_threshold: f32,\n    rules: PropFirmRiskRules,\n) -> Result<Vec<PropFirmRiskValidationArtifactFile>> {"
discovery = replace_exact(
    discovery,
    prop_signature,
    prop_replacement,
    label="prop-firm explicit gate API",
)

# The held-out tail uses the exact value returned by the in-sample GA.
discovery = replace_exact(
    discovery,
    "    match compute_discovery_forward_test_artifacts(\n        &result.portfolio,\n        &result.effective_feature_names,\n        &tail_features,\n        &tail_ohlcv,\n        config,\n    ) {",
    "    match compute_discovery_forward_test_artifacts_with_smc_gate(\n        &result.portfolio,\n        &result.effective_feature_names,\n        &tail_features,\n        &tail_ohlcv,\n        config,\n        result.effective_smc_gate_threshold,\n    ) {",
    label="holdout forward gate",
)
discovery = replace_exact(
    discovery,
    "    match compute_discovery_prop_firm_artifacts(\n        &result.portfolio,\n        &result.effective_feature_names,\n        &tail_features,\n        &tail_ohlcv,\n        config,\n        prop_firm_rules,\n    ) {",
    "    match compute_discovery_prop_firm_artifacts_with_smc_gate(\n        &result.portfolio,\n        &result.effective_feature_names,\n        &tail_features,\n        &tail_ohlcv,\n        config,\n        result.effective_smc_gate_threshold,\n        prop_firm_rules,\n    ) {",
    label="holdout prop-firm gate",
)

# Replace only the real production constructor inside finalize; test fixtures
# retain the explicit NaN sentinel inserted above.
finalize_start = discovery.index("fn finalize_candidates_with_progress")
finalize_end = discovery.index("\nfn candidate_truncation_limit", finalize_start)
finalize_region = discovery[finalize_start:finalize_end]
actual = finalize_region.count("effective_smc_gate_threshold: f32::NAN")
if actual != 1:
    raise RuntimeError(f"finalize DiscoveryResult gate: expected one sentinel, found {actual}")
finalize_region = finalize_region.replace(
    "effective_smc_gate_threshold: f32::NAN",
    "effective_smc_gate_threshold",
)
discovery = discovery[:finalize_start] + finalize_region + discovery[finalize_end:]

write(discovery_path, discovery)

# Document the verified migration commit name for the workflow bot.
(ROOT / ".github/agent-patches/commit-message.txt").write_text(
    "fix(discovery): preserve final SMC gate across validation\n",
    encoding="utf-8",
)

print(
    f"queued SMC migration: {result_literals} SearchResult constructors, "
    f"{changed_discovery_literals} DiscoveryResult literals"
)
