//! Prints the honest goal-report frontier for a few edge profiles, so the
//! operator can see "reach 50k, when and at what risk" in concrete numbers
//! before any card run. These are ILLUSTRATIVE edges (fixed +2R wins / -1R
//! losses at a given win rate); the real run feeds the portfolio's actual,
//! cost-charged R-multiples into the exact same function.
//!
//! Run: cargo run -p neoethos-search --example goal_frontier_demo

use neoethos_search::goal_report::{build_report, DEFAULT_RISK_LEVELS};

fn synthetic_edge(win_rate_pct: usize, reward_r: f64) -> Vec<f64> {
    let mut v = Vec::with_capacity(100);
    for _ in 0..win_rate_pct {
        v.push(reward_r);
    }
    for _ in 0..(100 - win_rate_pct) {
        v.push(-1.0);
    }
    v
}

fn main() {
    let horizon = 180.0;
    let trades_per_day = 2.0;

    let edges = [
        ("STRONG 2RR edge, 48% WR", synthetic_edge(48, 2.0)),
        ("GOOD 2RR edge, 45% WR", synthetic_edge(45, 2.0)),
        ("MARGINAL 2RR edge, 40% WR", synthetic_edge(40, 2.0)),
        ("OPERATOR'S OLD BROKEN edge, 44% WR @ 0.8R", synthetic_edge(44, 0.8)),
    ];

    // (start, target, label). 100->50k is x500; 1000->200k is x200 (easier).
    let goals = [
        (100.0, 50_000.0, "GOAL: 100 -> 50k (x500)"),
        (1000.0, 200_000.0, "GOAL: 1000 -> 200k (x200)"),
    ];

    for (start, target, glabel) in goals {
        println!("\n############## {glabel} ##############");
        for (name, r) in &edges {
            let rep = build_report(
                r,
                start,
                target,
                horizon,
                trades_per_day,
                DEFAULT_RISK_LEVELS,
                0xC0FFEE,
            );
            println!("\n=== {name} ===");
            print!("{}", rep.render());
        }
    }
}
