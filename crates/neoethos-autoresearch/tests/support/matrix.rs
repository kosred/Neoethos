use neoethos_search::trial_returns::{TrialReturnMatrix, TrialReturnRow, encode};

const PERIODS: usize = 24;
pub const TRIALS: usize = 40;
const FIRST_MONTH_KEY: i64 = 24_300;

fn month_keys() -> Vec<i64> {
    (0..PERIODS as i64)
        .map(|month| FIRST_MONTH_KEY + month)
        .collect()
}

fn champion_returns(lift: f64) -> Vec<f64> {
    (0..PERIODS)
        .map(|period| {
            if period % 2 == 0 {
                0.03 * lift
            } else {
                0.02 * lift
            }
        })
        .collect()
}

fn pack_returns(index: usize, drift: f64, offset: f64) -> Vec<f64> {
    (0..PERIODS)
        .map(|period| {
            let swing = if period % 2 == 0 { 0.01 } else { -0.01 };
            swing + drift * index as f64 + offset
        })
        .collect()
}

/// Build a deterministic trial-return matrix for the pure attribution tests.
/// This fixture never enters a financial evaluation or promotion path.
pub fn matrix_for(salt: usize) -> (Vec<u8>, Vec<f64>) {
    let lift = 1.0 + 0.01 * (salt % 7) as f64;
    let mut rows = Vec::with_capacity(TRIALS);
    let champion = champion_returns(lift);

    rows.push(TrialReturnRow {
        candidate_index: 0,
        strategy_id: format!("champ-{salt}"),
        returns: champion.clone(),
        trades_outside_grid: 0,
    });
    for index in 1..TRIALS {
        rows.push(TrialReturnRow {
            candidate_index: index,
            strategy_id: format!("pack-{salt}-{index}"),
            returns: pack_returns(index, 0.000_1, 0.0),
            trades_outside_grid: 0,
        });
    }

    let matrix = TrialReturnMatrix {
        period_keys: month_keys(),
        initial_balance: 10_000.0,
        rows,
    };
    let selected: Vec<&TrialReturnRow> = matrix.rows.iter().collect();
    (encode(&matrix, &selected), champion)
}
