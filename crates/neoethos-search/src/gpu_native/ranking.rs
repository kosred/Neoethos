//! Canonical integer ranking and semantic gene serialization.

use crate::genetic::Gene;
use neoethos_core::utils::fnv1a64;
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankQuantizationPolicy {
    pub fitness_scale: i64,
    pub ratio_scale: i64,
    pub drawdown_scale: i64,
    pub rate_scale: i64,
}

impl RankQuantizationPolicy {
    pub const CANONICAL: Self = Self {
        fitness_scale: 1_000_000,
        ratio_scale: 1_000_000,
        drawdown_scale: 1_000_000_000,
        rate_scale: 1_000_000,
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankKey {
    pub primary_fitness: i64,
    pub sharpe: i64,
    pub profit_factor: i64,
    /// Negated so lower drawdown sorts before higher drawdown.
    pub inverse_drawdown: i64,
    pub win_rate: i64,
    pub trade_count: u64,
    pub semantic_gene_hash: u64,
    pub canonical_gene_bytes: Vec<u8>,
    pub stable_trial_id: u64,
}

impl RankKey {
    pub fn from_gene(
        gene: &Gene,
        stable_trial_id: u64,
        policy: RankQuantizationPolicy,
    ) -> Result<Self, RankKeyError> {
        let canonical_gene_bytes = canonical_gene_bytes(gene)?;
        Ok(Self {
            primary_fitness: quantize_f64("fitness", gene.fitness, policy.fitness_scale)?,
            sharpe: quantize_f64("sharpe_ratio", gene.sharpe_ratio, policy.ratio_scale)?,
            profit_factor: quantize_f64(
                "profit_factor",
                gene.profit_factor,
                policy.ratio_scale,
            )?,
            inverse_drawdown: quantize_f64(
                "max_drawdown",
                -gene.max_drawdown,
                policy.drawdown_scale,
            )?,
            win_rate: quantize_f64("win_rate", gene.win_rate, policy.rate_scale)?,
            trade_count: gene.trades_count as u64,
            semantic_gene_hash: fnv1a64(&canonical_gene_bytes),
            canonical_gene_bytes,
            stable_trial_id,
        })
    }

    /// Best-first canonical ordering used by both CPU and GPU selection paths.
    pub fn cmp_best_first(&self, other: &Self) -> Ordering {
        other
            .primary_fitness
            .cmp(&self.primary_fitness)
            .then_with(|| other.sharpe.cmp(&self.sharpe))
            .then_with(|| other.profit_factor.cmp(&self.profit_factor))
            .then_with(|| other.inverse_drawdown.cmp(&self.inverse_drawdown))
            .then_with(|| other.win_rate.cmp(&self.win_rate))
            .then_with(|| other.trade_count.cmp(&self.trade_count))
            .then_with(|| self.semantic_gene_hash.cmp(&other.semantic_gene_hash))
            .then_with(|| self.canonical_gene_bytes.cmp(&other.canonical_gene_bytes))
            .then_with(|| self.stable_trial_id.cmp(&other.stable_trial_id))
    }
}

pub fn canonical_gene_bytes(gene: &Gene) -> Result<Vec<u8>, RankKeyError> {
    if gene.indices.len() != gene.weights.len() {
        return Err(RankKeyError::ShapeMismatch {
            indices: gene.indices.len(),
            weights: gene.weights.len(),
        });
    }

    let mut terms = Vec::with_capacity(gene.indices.len());
    for (index, weight) in gene.indices.iter().copied().zip(gene.weights.iter().copied()) {
        terms.push((index as u64, canonical_f32("weight", weight)?));
    }
    terms.sort_unstable_by_key(|(index, weight_bits)| (*index, *weight_bits));

    let mut out = Vec::new();
    push_u32(&mut out, terms.len() as u32);
    for (index, weight_bits) in terms {
        push_u64(&mut out, index);
        push_u32(&mut out, weight_bits);
    }

    push_u32(
        &mut out,
        canonical_f32("long_threshold", gene.long_threshold)?,
    );
    push_u32(
        &mut out,
        canonical_f32("short_threshold", gene.short_threshold)?,
    );

    let flags = [
        gene.use_ob,
        gene.use_fvg,
        gene.use_liq_sweep,
        gene.mtf_confirmation,
        gene.use_premium_discount,
        gene.use_inducement,
        gene.use_bos,
        gene.use_choch,
        gene.use_eqh,
        gene.use_eql,
        gene.use_displacement,
    ];
    out.extend(flags.map(u8::from));

    push_u64(&mut out, canonical_f64("tp_pips", gene.tp_pips)?);
    push_u64(&mut out, canonical_f64("sl_pips", gene.sl_pips)?);
    push_u64(
        &mut out,
        canonical_f64("stop_vol_mult", gene.stop_vol_mult)?,
    );
    Ok(out)
}

fn quantize_f64(field: &'static str, value: f64, scale: i64) -> Result<i64, RankKeyError> {
    if !value.is_finite() {
        return Err(RankKeyError::NonFinite { field });
    }
    let scaled = value * scale as f64;
    if scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
        return Err(RankKeyError::OutOfRange { field });
    }
    Ok(scaled.round() as i64)
}

fn canonical_f32(field: &'static str, value: f32) -> Result<u32, RankKeyError> {
    if !value.is_finite() {
        return Err(RankKeyError::NonFinite { field });
    }
    Ok(if value == 0.0 { 0.0_f32 } else { value }.to_bits())
}

fn canonical_f64(field: &'static str, value: f64) -> Result<u64, RankKeyError> {
    if !value.is_finite() {
        return Err(RankKeyError::NonFinite { field });
    }
    Ok(if value == 0.0 { 0.0_f64 } else { value }.to_bits())
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RankKeyError {
    ShapeMismatch { indices: usize, weights: usize },
    NonFinite { field: &'static str },
    OutOfRange { field: &'static str },
}

impl fmt::Display for RankKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { indices, weights } => write!(
                f,
                "gene shape mismatch: {indices} indicator indices but {weights} weights"
            ),
            Self::NonFinite { field } => write!(f, "non-finite value in canonical rank field {field}"),
            Self::OutOfRange { field } => write!(f, "quantized rank field {field} exceeds i64 range"),
        }
    }
}

impl Error for RankKeyError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn gene() -> Gene {
        Gene {
            indices: vec![3, 1],
            weights: vec![-0.0, 0.5],
            long_threshold: 0.25,
            short_threshold: -0.25,
            fitness: 10.0,
            sharpe_ratio: 1.5,
            profit_factor: 1.8,
            max_drawdown: 0.05,
            win_rate: 0.55,
            trades_count: 40,
            tp_pips: 30.0,
            sl_pips: 20.0,
            stop_vol_mult: 0.0,
            ..Gene::default()
        }
    }

    #[test]
    fn indicator_order_and_negative_zero_are_canonical() {
        let first = gene();
        let mut second = first.clone();
        second.indices = vec![1, 3];
        second.weights = vec![0.5, 0.0];
        assert_eq!(
            canonical_gene_bytes(&first).unwrap(),
            canonical_gene_bytes(&second).unwrap()
        );
    }

    #[test]
    fn non_finite_semantic_values_are_rejected() {
        let mut candidate = gene();
        candidate.long_threshold = f32::NAN;
        assert!(matches!(
            canonical_gene_bytes(&candidate),
            Err(RankKeyError::NonFinite {
                field: "long_threshold"
            })
        ));
    }

    #[test]
    fn stable_trial_id_resolves_complete_semantic_collision() {
        let first = RankKey::from_gene(&gene(), 1, RankQuantizationPolicy::CANONICAL).unwrap();
        let second = RankKey::from_gene(&gene(), 2, RankQuantizationPolicy::CANONICAL).unwrap();
        assert_eq!(first.semantic_gene_hash, second.semantic_gene_hash);
        assert_eq!(first.canonical_gene_bytes, second.canonical_gene_bytes);
        assert_eq!(first.cmp_best_first(&second), Ordering::Less);
    }

    #[test]
    fn integer_rank_fields_determine_exact_order() {
        let first = RankKey::from_gene(&gene(), 1, RankQuantizationPolicy::CANONICAL).unwrap();
        let mut stronger_gene = gene();
        stronger_gene.fitness += 0.01;
        let stronger =
            RankKey::from_gene(&stronger_gene, 2, RankQuantizationPolicy::CANONICAL).unwrap();
        assert_eq!(stronger.cmp_best_first(&first), Ordering::Less);
    }
}
