//! Offline learning FROM live experience — the follow-up the experience store
//! was built for.
//!
//! Trains an online expert (AdaptiveGradientBooster) to predict win/loss from
//! the EXACT entry feature rows the live engine acted on, and answers the only
//! question that matters honestly: **do live outcomes contain learnable
//! signal beyond the majority class?** — measured on a strictly TIME-ORDERED
//! holdout (last 20% of records, never shuffled: no leakage from the future).
//!
//! DISCIPLINE (operator + house rule): this trainer NEVER influences live
//! trading. It produces a report; wiring any learned model into live decisions
//! is a separate, explicitly-validated step (same OOS bar as everything else).
//! Until then, this is the honest mirror: "your live experience is/them isn't
//! predictable yet — keep collecting".

use anyhow::{Context, Result};
use serde::Serialize;

use neoethos_models::AdaptiveGradientBooster;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GroupReport {
    /// The portfolio whose feature space these records share.
    pub portfolio: String,
    pub records: usize,
    pub train_n: usize,
    pub test_n: usize,
    /// Fraction of wins in the TEST slice (the bar to beat).
    pub baseline_pct: f64,
    /// Out-of-sample accuracy of the learned model on the TEST slice.
    pub oos_accuracy_pct: f64,
    /// oos_accuracy − max(baseline, 1−baseline): positive = learnable signal.
    pub edge_pct: f64,
    pub verdict: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExperienceTrainReport {
    pub total_records: usize,
    pub usable_records: usize,
    pub groups: Vec<GroupReport>,
    pub note: String,
}

const MIN_GROUP_RECORDS: usize = 40;

/// BLOCKING. Load the experience store, train per portfolio group, report
/// time-ordered OOS accuracy vs the majority baseline.
pub fn train_from_experience() -> Result<ExperienceTrainReport> {
    let path = neoethos_core::Settings::from_yaml(&crate::server::state::current_config_path())
        .context("config.yaml not loadable")?
        .system
        .data_dir
        .join("experience")
        .join("live_experience.jsonl");
    let raw = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "no live experience yet at {} — run the autopilot; every closed \
             trade adds a record",
            path.display()
        )
    })?;

    // (portfolio, entry_ts, features, win)
    let mut rows: Vec<(String, i64, Vec<f64>, bool)> = Vec::new();
    let mut total = 0usize;
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        total += 1;
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        let Some(net) = v.get("netProfit").and_then(|x| x.as_f64()) else { continue };
        let feats: Vec<f64> = v
            .get("features")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|f| f.as_f64()).collect())
            .unwrap_or_default();
        if feats.is_empty() || feats.iter().any(|f| !f.is_finite()) {
            continue;
        }
        let portfolio = v
            .get("portfolioPath")
            .and_then(|x| x.as_str())
            .unwrap_or("?")
            .to_string();
        let ts = v.get("entryTsMs").and_then(|x| x.as_i64()).unwrap_or(0);
        rows.push((portfolio, ts, feats, net > 0.0));
    }

    // Group by portfolio (each has its own feature space).
    let mut groups: std::collections::BTreeMap<String, Vec<(i64, Vec<f64>, bool)>> =
        std::collections::BTreeMap::new();
    for (p, ts, f, w) in rows.iter().cloned() {
        groups.entry(p).or_default().push((ts, f, w));
    }

    let mut reports = Vec::new();
    for (portfolio, mut recs) in groups {
        // Strict TIME order — the holdout is the FUTURE, never a shuffle.
        recs.sort_by_key(|r| r.0);
        // Uniform feature length inside the group (schema drift guard).
        let flen = recs.first().map(|r| r.1.len()).unwrap_or(0);
        recs.retain(|r| r.1.len() == flen);
        let n = recs.len();
        if n < MIN_GROUP_RECORDS {
            reports.push(GroupReport {
                portfolio,
                records: n,
                train_n: 0,
                test_n: 0,
                baseline_pct: 0.0,
                oos_accuracy_pct: 0.0,
                edge_pct: 0.0,
                verdict: format!("collecting — needs ≥{MIN_GROUP_RECORDS} closed trades"),
            });
            continue;
        }
        let split = (n as f64 * 0.8) as usize;
        let (train, test) = recs.split_at(split);

        let mut model = AdaptiveGradientBooster::new();
        for (_, x, w) in train {
            let _ = model.learn_one(x.clone(), if *w { 1.0 } else { 0.0 });
        }
        let mut correct = 0usize;
        let mut wins_in_test = 0usize;
        for (_, x, w) in test {
            if *w {
                wins_in_test += 1;
            }
            let pred = model.predict_one(x).unwrap_or(0.5) > 0.5;
            if pred == *w {
                correct += 1;
            }
        }
        let test_n = test.len();
        let baseline = wins_in_test as f64 / test_n as f64;
        let majority = baseline.max(1.0 - baseline) * 100.0;
        let acc = correct as f64 / test_n as f64 * 100.0;
        let edge = acc - majority;
        let verdict = if edge > 5.0 {
            "LEARNABLE: live outcomes carry signal beyond the majority class — \
             a validated exit/meta-label layer is worth building"
                .to_string()
        } else if edge > 0.0 {
            "weak signal — keep collecting before drawing conclusions".to_string()
        } else {
            "no learnable signal yet (model ≤ majority baseline) — honest answer: \
             the entry features don't predict live win/loss so far"
                .to_string()
        };
        reports.push(GroupReport {
            portfolio,
            records: n,
            train_n: split,
            test_n,
            baseline_pct: baseline * 100.0,
            oos_accuracy_pct: acc,
            edge_pct: edge,
            verdict,
        });
    }

    let usable = rows.len();
    Ok(ExperienceTrainReport {
        total_records: total,
        usable_records: usable,
        groups: reports,
        note: "Report only — nothing here influences live trading. A learned layer \
               must pass the same OOS discipline as every strategy before it may."
            .to_string(),
    })
}
