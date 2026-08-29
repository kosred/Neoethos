//! Backend-independent Search objective authority.
//!
//! CUDA and CPU oracles bind the semantic identity below. The functions keep
//! the current named-search formulas in one dependency that has no GPU runtime.

pub const PROPFIRM_GA_FITNESS_V4_SEMANTICS: &str = concat!(
    "neoethos.search.objective.propfirm-v4;",
    "slots=net,sharpe,peak,max-dd,win-rate,pf,expectancy,monthly-hit,trades,consistency,max-daily-dd;",
    "zero-trades=-100;nonfinite-sharpe=-inf"
);

pub const RISKY_GA_FITNESS_GROWTH_V5_SEMANTICS: &str = concat!(
    "neoethos.search.objective.risky-growth-v5;",
    "half-kelly;fraction-cap=0.25;pf-cap=10;win-rate-cap=0.99;",
    "zero-trades=-100;nonfinite-sharpe=-inf"
);

#[inline]
fn trades_confidence(trades: f64) -> f64 {
    (trades.sqrt() / 10.0).min(1.0)
}

#[inline]
fn ga_pf_component(profit_factor: f64) -> f64 {
    if profit_factor >= 1.0 {
        ((profit_factor - 1.0) * 0.5).min(1.5)
    } else {
        -(1.0 / profit_factor.max(0.1))
    }
}

/// Current PropFirm GA objective, scoring version 4.
pub fn score_prop_firm_ga_fitness_v4(metrics: &[f64; 11]) -> f64 {
    let net = metrics[0];
    let sharpe = metrics[1];
    let max_drawdown = metrics[3];
    let win_rate = metrics[4];
    let profit_factor = metrics[5];
    let monthly_hit = metrics[7];
    let trades = metrics[8];
    let consistency = metrics[9];
    let max_daily_drawdown = metrics[10];
    if !sharpe.is_finite() {
        return f64::NEG_INFINITY;
    }
    if trades < 1.0 {
        return -100.0;
    }
    let activity_multiplier = 0.3 + 0.7 * (trades / 30.0).clamp(0.0, 1.0);
    let confidence = trades_confidence(trades);
    let hit = monthly_hit.clamp(0.0, 1.0) * 0.45;
    let net_return = (net / 20_000.0).clamp(-2.0, 2.0) * 0.15;
    let sharpe_score = sharpe.clamp(-2.0, 4.0) * confidence * 0.10;
    let consistency_score = consistency.clamp(0.0, 1.0) * 0.10;
    let profit_factor_score =
        ga_pf_component(profit_factor) * if profit_factor >= 1.0 { 0.15 } else { 0.25 };
    let win_rate_score = ((win_rate.clamp(0.0, 1.0) - 0.45) * 2.0).clamp(0.0, 0.5) * 0.10;
    let drawdown = (max_drawdown.max(0.0) * 15.0).min(5.0);
    let daily_drawdown = max_daily_drawdown.clamp(0.0, 1.0) * 10.0;
    (hit + net_return + sharpe_score + consistency_score + profit_factor_score + win_rate_score)
        * activity_multiplier
        - drawdown
        - daily_drawdown
}

/// Current Risky GA half-Kelly objective, scoring version 5.
pub fn score_risky_ga_fitness_growth_v5(metrics: &[f64; 11]) -> f64 {
    let net = metrics[0];
    let sharpe = metrics[1];
    let win_rate = metrics[4];
    let profit_factor = metrics[5];
    let trades = metrics[8];
    if !sharpe.is_finite() {
        return f64::NEG_INFINITY;
    }
    if trades < 1.0 {
        return -100.0;
    }
    let p = win_rate.clamp(0.0, 0.99);
    let pf = profit_factor.clamp(0.0, 10.0);
    let f_star = if pf > 1.0 && p > 0.0 {
        p * (pf - 1.0) / pf
    } else {
        0.0
    };
    let f = (f_star * 0.5).clamp(0.0, 0.25);
    let rr = if p > 0.0 { pf * (1.0 - p) / p } else { 0.0 };
    let growth_per_trade = if f > 0.0 && rr > 0.0 {
        p * (1.0 + rr * f).ln() + (1.0 - p) * (1.0 - f).ln()
    } else {
        0.0
    };
    let edge_gradient = (pf - 1.0).clamp(-1.0, 0.0) * 0.05
        + (p - 0.5).clamp(-0.5, 0.0) * 0.05
        + (net / 20_000.0).clamp(-2.0, 0.0) * 0.01;
    growth_per_trade * trades * 10.0 + edge_gradient
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_objectives_keep_named_pins() {
        let mut metrics = [0.0; 11];
        metrics[0] = 1_000.0;
        metrics[1] = 2.0;
        metrics[3] = 0.05;
        metrics[4] = 0.60;
        metrics[5] = 1.8;
        metrics[8] = 100.0;
        metrics[9] = 0.70;
        assert!((score_prop_firm_ga_fitness_v4(&metrics) - -0.3825).abs() < 1.0e-9);
        assert!(score_risky_ga_fitness_growth_v5(&metrics).is_finite());
    }
}
