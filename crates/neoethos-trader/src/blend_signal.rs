//! Phase 4 / v0.5 ML-integration Stage 3 — gene-dominant ML meta-gate blend.
//!
//! López de Prado meta-labeling: the **genes decide DIRECTION** (the
//! OOS-validated edge, untouched), and the **ML ensemble decides BET/SIZE** —
//! it can only SHRINK conviction or VETO a trade, never flip Long↔Short and
//! never manufacture a trade from Flat. This makes "do not degrade the
//! validated gene edge" a STRUCTURAL invariant, not a hope.
//!
//! The blend CORE here (math + [`BlendedSignalEngine`] + invariants) is
//! always compiled and depends on NOTHING heavy — [`MlDecision`] is a local
//! mirror of `neoethos_models::ensemble_inference::EnsembleDecision`. Only the
//! actual ensemble-loading replay path (`data_replay::replay_blend_from_dir`,
//! behind the `ml-blend` feature) pulls in the ML stack and converts the real
//! decisions into [`MlDecision`].

use std::collections::HashMap;

use crate::contracts::{Direction, LiveBar, PortfolioEntry, Signal, SignalEngine, SignalSource};

/// Per-bar ML decision the blend consumes. Mirror of
/// `neoethos_models::ensemble_inference::EnsembleDecision` kept LOCAL so the
/// safety-critical blend core compiles without the heavy ML crates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlDecision {
    /// `[p_neutral, p_buy, p_sell]` from the ensemble's directional voters.
    pub dir_probs: [f64; 3],
    /// Regime gate ∈ [0,1] (1.0 = no gate; → 0 shrinks/vetoes in a range/
    /// disagreeing regime).
    pub regime_gate: f64,
    /// Anomaly scale ∈ [0,1] (1.0 = no penalty; 0.0 = hard veto on an extreme
    /// anomaly).
    pub anomaly_scale: f64,
}

impl MlDecision {
    /// Neutral — the ensemble abstains (warmup/NaN rows): no directional lean,
    /// no gate, no veto.
    pub fn neutral() -> Self {
        Self {
            dir_probs: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            regime_gate: 1.0,
            anomaly_scale: 1.0,
        }
    }
}

/// How the gene direction and the ML decision combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Production default — ML is NOT consulted; behaviour is byte-identical to
    /// the gene-only [`crate::gene_signal::PrecomputedSignalEngine`].
    GenesOnly,
    /// Meta-label gate: keep gene direction, scale size by ML agreement × gates,
    /// and VETO to Flat when ML disagrees hard (`p_side < veto_below`) or a gate
    /// collapses.
    MlConfirm,
    /// Soft size only: keep gene direction, scale size by ML agreement × gates;
    /// never veto on ML disagreement (but a hard regime/anomaly collapse still
    /// skips the trade, since the sizing floor would otherwise open min volume).
    MlScale,
}

/// Default floor on the ML agreement term (audit #232).
///
/// This number and [`DEFAULT_BLEND_VETO_BELOW`] SCALE EVERY ENTRY'S RISK on
/// the live path (`live_trading.rs` builds a `BlendConfig` and multiplies the
/// gene's size by the resulting confidence). Until 2026-08-09 both were bare
/// literals inside `Default` with **no config recipient anywhere**, so the
/// operator could not see them, let alone change them. They are named here so
/// they are greppable and so a config field can bind to them without moving
/// the number; the values themselves are UNCHANGED.
///
/// WIRING (2026-08-10). Every production construction site now goes through
/// [`BlendConfig::from_config_values`] rather than a struct literal:
/// * `neoethos-app/src/app_services/live_trading.rs` — built ONCE at engine
///   start (`live_blend_cfg`), consumed by `blend_decision` on every entry bar.
/// * `neoethos-cli/src/main.rs` `cmd_trader_replay` — `--gate-floor` /
///   `--veto-below` are handed to the constructor instead of being written into
///   the fields, so the CLI can no longer build a blend the constructor rejects.
///
/// The YAML recipient (`models.blend_gate_floor` / `models.blend_veto_below`)
/// still does not
/// exist — `neoethos-core/src/config.rs` is owned by another workflow this wave,
/// and the exact field list is written down in
/// `docs/pending-edits-forbidden-territory.md`. The live readers
/// (`operator_blend_gate_floor` / `operator_blend_veto_below` in
/// `live_trading.rs`) therefore pass `None` today and land on these defaults.
/// The CLI flags are already live operator input.
pub const DEFAULT_BLEND_GATE_FLOOR: f64 = 0.34;

/// Default effective-multiplier floor below which the trade is SKIPPED.
/// See [`DEFAULT_BLEND_GATE_FLOOR`] for why this is a named constant.
pub const DEFAULT_BLEND_VETO_BELOW: f64 = 0.15;

/// Blend tunables. Defaults keep the gene edge dominant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendConfig {
    pub mode: BlendMode,
    /// Floor on the ML agreement term so a healthy gene bar always trades a
    /// meaningful size even when ML is only lukewarm.
    /// Default [`DEFAULT_BLEND_GATE_FLOOR`].
    pub gate_floor: f64,
    /// Below this effective multiplier the trade is SKIPPED (set Flat, not just
    /// confidence 0 — the sizing floor would otherwise open min volume).
    /// In `MlConfirm` also vetoes when the raw ML `p_side` is below it.
    /// Default [`DEFAULT_BLEND_VETO_BELOW`].
    pub veto_below: f64,
}

impl Default for BlendConfig {
    fn default() -> Self {
        Self {
            mode: BlendMode::GenesOnly,
            gate_floor: DEFAULT_BLEND_GATE_FLOOR,
            veto_below: DEFAULT_BLEND_VETO_BELOW,
        }
    }
}

impl BlendConfig {
    /// Build a config from OPTIONAL operator-supplied values — the ONLY
    /// sanctioned way production code builds a [`BlendConfig`] with non-default
    /// multipliers (audit #232). Callers: `live_trading.rs` (live sizing) and
    /// `neoethos-cli` `cmd_trader_replay` (`--gate-floor` / `--veto-below`).
    /// Writing `gate_floor` / `veto_below` directly bypasses every refusal
    /// below; the struct literal form is reserved for tests.
    ///
    /// SAFETY POSTURE: a value that is absent, non-finite, or outside `[0,1]`
    /// does NOT silently become something else. It falls back to the shipped
    /// default and the fallback is logged at `warn` with both numbers, because
    /// these two multipliers scale real position size. `veto_below` above
    /// `gate_floor` would veto every trade the floor was meant to keep
    /// tradeable, so that combination is also refused back to the defaults and
    /// logged — fail toward the validated behaviour, never toward a novel one.
    pub fn from_config_values(
        mode: BlendMode,
        gate_floor: Option<f64>,
        veto_below: Option<f64>,
    ) -> Self {
        fn resolve(name: &'static str, given: Option<f64>, fallback: f64) -> f64 {
            match given {
                Some(v) if v.is_finite() && (0.0..=1.0).contains(&v) => v,
                Some(v) => {
                    tracing::warn!(
                        target: "neoethos_trader::blend",
                        knob = name,
                        configured = v,
                        used = fallback,
                        "blend knob outside [0,1] or non-finite — REFUSED, using the \
                         shipped default. This multiplier scales every entry's size."
                    );
                    fallback
                }
                None => fallback,
            }
        }
        let gate_floor = resolve("gate_floor", gate_floor, DEFAULT_BLEND_GATE_FLOOR);
        let veto_below = resolve("veto_below", veto_below, DEFAULT_BLEND_VETO_BELOW);
        if veto_below > gate_floor {
            tracing::warn!(
                target: "neoethos_trader::blend",
                configured_gate_floor = gate_floor,
                configured_veto_below = veto_below,
                "veto_below exceeds gate_floor — every floored bar would be vetoed. \
                 REFUSED: reverting BOTH to the shipped defaults."
            );
            return Self {
                mode,
                gate_floor: DEFAULT_BLEND_GATE_FLOOR,
                veto_below: DEFAULT_BLEND_VETO_BELOW,
            };
        }
        Self {
            mode,
            gate_floor,
            veto_below,
        }
    }
}

/// Pure blend math (no I/O, fully unit-testable). Given the gene's direction and
/// one [`MlDecision`], return the (possibly vetoed) direction + confidence.
///
/// INVARIANTS (tested):
/// - Flat gene ⇒ Flat out (ML never manufactures a trade).
/// - out direction ∈ {gene direction, Flat} (ML never flips Long↔Short).
/// - confidence ∈ [0,1]; `gate_floor` keeps a healthy bar tradeable; a hard
///   regime/anomaly collapse (or, in MlConfirm, ML disagreement) ⇒ Flat.
pub fn blend_decision(dir: Direction, ml: &MlDecision, cfg: &BlendConfig) -> (Direction, f64) {
    let p_side = match dir {
        Direction::Long => ml.dir_probs[1] as f64,
        Direction::Short => ml.dir_probs[2] as f64,
        Direction::Flat => return (Direction::Flat, 0.0),
    };
    let g = (ml.regime_gate as f64).clamp(0.0, 1.0);
    let s = (ml.anomaly_scale as f64).clamp(0.0, 1.0);
    // ML agreement floored (so a healthy gene bar always trades a meaningful
    // size); the regime/anomaly gates are applied OUTSIDE the floor so a hard
    // veto (g≈0 or s≈0) can still drive the multiplier to ~0.
    let agreement = p_side.clamp(cfg.gate_floor, 1.0);
    let m = (agreement * g * s).clamp(0.0, 1.0);

    let disagree_veto = matches!(cfg.mode, BlendMode::MlConfirm) && p_side < cfg.veto_below;
    if disagree_veto || m < cfg.veto_below {
        // Skip the trade entirely — Flat, NOT confidence 0 (the DecisionEngine
        // floors sizing to min_volume, so confidence 0 would still open a trade).
        (Direction::Flat, 0.0)
    } else {
        (dir, m)
    }
}

/// A [`SignalEngine`] that serves precomputed gene directions, optionally gated
/// by precomputed per-bar [`MlDecision`]s. With `mode == GenesOnly` or no ML
/// vector for the symbol, it is byte-identical to
/// [`crate::gene_signal::PrecomputedSignalEngine`] (confidence 1.0 directional /
/// 0.0 flat, `SignalSource::Strategy`) — the hard fallback when the ensemble is
/// absent / failed to load / column-mismatched.
pub struct BlendedSignalEngine {
    per_symbol_dir: HashMap<String, Vec<Direction>>,
    per_symbol_ml: HashMap<String, Vec<MlDecision>>,
    /// Per-bar STRATEGY brackets in pips (audit #226). Empty ⇒ no bracket, and
    /// the DecisionEngine's synthetic stop applies (and says so).
    per_symbol_sl: HashMap<String, Vec<f64>>,
    per_symbol_tp: HashMap<String, Vec<f64>>,
    cfg: BlendConfig,
    cursors: HashMap<String, usize>,
}

impl BlendedSignalEngine {
    /// Gene-only engine (no ML) — byte-identical to `PrecomputedSignalEngine`.
    pub fn genes_only(symbol: &str, directions: Vec<Direction>) -> Self {
        let mut per_symbol_dir = HashMap::new();
        per_symbol_dir.insert(symbol.to_string(), directions);
        Self {
            per_symbol_dir,
            per_symbol_ml: HashMap::new(),
            per_symbol_sl: HashMap::new(),
            per_symbol_tp: HashMap::new(),
            cfg: BlendConfig::default(),
            cursors: HashMap::new(),
        }
    }

    /// Blended engine: gene directions gated by per-bar ML decisions.
    /// `ml.len()` should equal `directions.len()`; missing entries fall back to
    /// the gene-only path for that bar (defensive).
    pub fn new(
        symbol: &str,
        directions: Vec<Direction>,
        ml: Vec<MlDecision>,
        cfg: BlendConfig,
    ) -> Self {
        let mut per_symbol_dir = HashMap::new();
        per_symbol_dir.insert(symbol.to_string(), directions);
        let mut per_symbol_ml = HashMap::new();
        per_symbol_ml.insert(symbol.to_string(), ml);
        Self {
            per_symbol_dir,
            per_symbol_ml,
            per_symbol_sl: HashMap::new(),
            per_symbol_tp: HashMap::new(),
            cfg,
            cursors: HashMap::new(),
        }
    }

    /// Attach the genes' OWN per-bar brackets (pips) to whatever this engine
    /// already serves. The ML gate can shrink or veto SIZE; it never touches
    /// the bracket, so the stop stays the one the gene was scored on.
    pub fn with_brackets(mut self, symbol: &str, sl_pips: Vec<f64>, tp_pips: Vec<f64>) -> Self {
        self.per_symbol_sl.insert(symbol.to_string(), sl_pips);
        self.per_symbol_tp.insert(symbol.to_string(), tp_pips);
        self
    }
}

impl SignalEngine for BlendedSignalEngine {
    fn evaluate(&mut self, entry: &PortfolioEntry, _window: &[LiveBar]) -> Signal {
        let cur = *self.cursors.get(&entry.symbol).unwrap_or(&0);
        let dir = self
            .per_symbol_dir
            .get(&entry.symbol)
            .and_then(|v| v.get(cur).copied())
            .unwrap_or(Direction::Flat);
        let ml = self
            .per_symbol_ml
            .get(&entry.symbol)
            .and_then(|v| v.get(cur).copied());
        let sl_pips = self
            .per_symbol_sl
            .get(&entry.symbol)
            .and_then(|v| v.get(cur).copied())
            .unwrap_or(0.0);
        let tp_pips = self
            .per_symbol_tp
            .get(&entry.symbol)
            .and_then(|v| v.get(cur).copied())
            .unwrap_or(0.0);
        self.cursors.insert(entry.symbol.clone(), cur + 1);

        match (self.cfg.mode, ml) {
            // Gene-only fallback — byte-identical to PrecomputedSignalEngine.
            (BlendMode::GenesOnly, _) | (_, None) => {
                let confidence = if dir == Direction::Flat { 0.0 } else { 1.0 };
                Signal {
                    symbol: entry.symbol.clone(),
                    dir,
                    confidence,
                    source: SignalSource::Strategy,
                    sl_pips,
                    tp_pips,
                }
            }
            (_, Some(decision)) => {
                let (out_dir, confidence) = blend_decision(dir, &decision, &self.cfg);
                Signal {
                    symbol: entry.symbol.clone(),
                    dir: out_dir,
                    confidence,
                    source: SignalSource::Blend,
                    sl_pips,
                    tp_pips,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{StrategySource, TradeMode};
    use crate::gene_signal::PrecomputedSignalEngine;

    fn entry() -> PortfolioEntry {
        PortfolioEntry {
            symbol: "EURUSD".to_string(),
            base_tf: "H1".to_string(),
            higher_tfs: Vec::new(),
            source: StrategySource::Gene {
                id: "x".to_string(),
            },
            mode: TradeMode::PropFirm,
        }
    }

    fn strong_buy() -> MlDecision {
        MlDecision {
            dir_probs: [0.05, 0.9, 0.05],
            regime_gate: 1.0,
            anomaly_scale: 1.0,
        }
    }
    fn strong_sell() -> MlDecision {
        MlDecision {
            dir_probs: [0.05, 0.05, 0.9],
            regime_gate: 1.0,
            anomaly_scale: 1.0,
        }
    }

    #[test]
    fn genes_only_is_byte_identical_to_precomputed() {
        let dirs = vec![
            Direction::Long,
            Direction::Flat,
            Direction::Short,
            Direction::Long,
        ];
        let mut blended = BlendedSignalEngine::genes_only("EURUSD", dirs.clone());
        let mut baseline = PrecomputedSignalEngine::new("EURUSD", dirs);
        let e = entry();
        for _ in 0..4 {
            let a = blended.evaluate(&e, &[]);
            let b = baseline.evaluate(&e, &[]);
            assert_eq!(a.dir, b.dir);
            assert_eq!(a.confidence, b.confidence);
            assert_eq!(a.source, b.source); // both SignalSource::Strategy
        }
    }

    #[test]
    fn ml_never_flips_direction() {
        // Gene says Long; ML screams sell. Output must be Long or Flat, NEVER Short.
        let cfg = BlendConfig {
            mode: BlendMode::MlConfirm,
            ..Default::default()
        };
        let mut eng =
            BlendedSignalEngine::new("EURUSD", vec![Direction::Long], vec![strong_sell()], cfg);
        let sig = eng.evaluate(&entry(), &[]);
        assert_ne!(
            sig.dir,
            Direction::Short,
            "ML must never flip the gene direction"
        );
        assert!(matches!(sig.dir, Direction::Long | Direction::Flat));
    }

    #[test]
    fn ml_never_creates_trade_from_flat() {
        let cfg = BlendConfig {
            mode: BlendMode::MlConfirm,
            ..Default::default()
        };
        let mut eng =
            BlendedSignalEngine::new("EURUSD", vec![Direction::Flat], vec![strong_buy()], cfg);
        let sig = eng.evaluate(&entry(), &[]);
        assert_eq!(
            sig.dir,
            Direction::Flat,
            "ML must never manufacture a trade from Flat"
        );
        assert_eq!(sig.confidence, 0.0);
    }

    #[test]
    fn gate_floor_keeps_a_healthy_gene_bar_tradeable() {
        // Lukewarm ML agreement (p_side 0.4), healthy gates -> trades at the floor.
        let cfg = BlendConfig {
            mode: BlendMode::MlConfirm,
            ..Default::default()
        };
        let lukewarm = MlDecision {
            dir_probs: [0.3, 0.4, 0.3],
            regime_gate: 1.0,
            anomaly_scale: 1.0,
        };
        let (dir, conf) = blend_decision(Direction::Long, &lukewarm, &cfg);
        assert_eq!(dir, Direction::Long);
        // tolerance accommodates the f32->f64 widening of dir_probs (0.4f32).
        assert!(
            (conf - 0.4).abs() < 1e-6,
            "expected agreement 0.4, got {conf}"
        );
    }

    #[test]
    fn hard_anomaly_veto_sets_flat_not_min_volume() {
        // Strong ML buy agreement, but anomaly_scale 0 -> hard veto -> Flat.
        let cfg = BlendConfig {
            mode: BlendMode::MlScale,
            ..Default::default()
        };
        let anomalous = MlDecision {
            dir_probs: [0.05, 0.9, 0.05],
            regime_gate: 1.0,
            anomaly_scale: 0.0,
        };
        let (dir, conf) = blend_decision(Direction::Long, &anomalous, &cfg);
        assert_eq!(
            dir,
            Direction::Flat,
            "hard anomaly veto must skip the trade"
        );
        assert_eq!(conf, 0.0);
    }

    /// Audit #232: an out-of-range or inverted operator value must NOT be
    /// applied — these two numbers scale every entry's size.
    #[test]
    fn from_config_values_refuses_bad_input_and_keeps_the_shipped_defaults() {
        let d = BlendConfig::default();

        let out_of_range =
            BlendConfig::from_config_values(BlendMode::MlScale, Some(1.5), Some(-0.2));
        assert_eq!(out_of_range.gate_floor, d.gate_floor);
        assert_eq!(out_of_range.veto_below, d.veto_below);

        let inverted = BlendConfig::from_config_values(BlendMode::MlScale, Some(0.10), Some(0.80));
        assert_eq!(
            inverted.gate_floor, d.gate_floor,
            "inverted pair must revert both"
        );
        assert_eq!(inverted.veto_below, d.veto_below);

        let good = BlendConfig::from_config_values(BlendMode::MlScale, Some(0.50), Some(0.20));
        assert_eq!(good.gate_floor, 0.50);
        assert_eq!(good.veto_below, 0.20);

        let unset = BlendConfig::from_config_values(BlendMode::GenesOnly, None, None);
        assert_eq!(unset.gate_floor, DEFAULT_BLEND_GATE_FLOOR);
        assert_eq!(unset.veto_below, DEFAULT_BLEND_VETO_BELOW);
    }

    /// The live path (`live_trading.rs`) and the CLI now build their
    /// `BlendConfig` through `from_config_values` instead of
    /// `BlendConfig { mode, ..Default::default() }`. That swap is only safe if
    /// the two forms are IDENTICAL when the operator has configured nothing —
    /// this pins it for every mode, so the 2026-08-10 rewiring cannot have
    /// moved a live position size by accident.
    #[test]
    fn unset_config_is_byte_identical_to_the_old_default_literal() {
        for mode in [
            BlendMode::GenesOnly,
            BlendMode::MlConfirm,
            BlendMode::MlScale,
        ] {
            let via_ctor = BlendConfig::from_config_values(mode, None, None);
            let via_literal = BlendConfig {
                mode,
                ..Default::default()
            };
            assert_eq!(
                via_ctor, via_literal,
                "from_config_values(None, None) must reproduce the shipped default for {mode:?}"
            );
        }
    }

    /// NaN is not "no value" — it is a value that would make every comparison
    /// in `blend_decision` false. It must be REFUSED, not propagated.
    #[test]
    fn non_finite_blend_knobs_are_refused() {
        let d = BlendConfig::default();
        let nan = BlendConfig::from_config_values(BlendMode::MlScale, Some(f64::NAN), None);
        assert_eq!(nan.gate_floor, d.gate_floor);
        let inf = BlendConfig::from_config_values(BlendMode::MlScale, None, Some(f64::INFINITY));
        assert_eq!(inf.veto_below, d.veto_below);
    }

    #[test]
    fn mlconfirm_vetoes_disagreement_but_mlscale_shrinks() {
        // ML disagrees with gene Long (p_buy 0.1 < veto_below 0.15).
        let disagree = MlDecision {
            dir_probs: [0.2, 0.1, 0.7],
            regime_gate: 1.0,
            anomaly_scale: 1.0,
        };
        let confirm = BlendConfig {
            mode: BlendMode::MlConfirm,
            ..Default::default()
        };
        let (d, _) = blend_decision(Direction::Long, &disagree, &confirm);
        assert_eq!(d, Direction::Flat, "MlConfirm vetoes on disagreement");

        let scale = BlendConfig {
            mode: BlendMode::MlScale,
            ..Default::default()
        };
        let (d, c) = blend_decision(Direction::Long, &disagree, &scale);
        assert_eq!(d, Direction::Long, "MlScale keeps direction, just shrinks");
        // agreement floored to gate_floor 0.34 * 1 * 1 = 0.34
        assert!((c - 0.34).abs() < 1e-9, "expected floored 0.34, got {c}");
    }
}
