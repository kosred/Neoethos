//! Experimental-only fitness percentile for genuine NeoEthos host GA.
//! Gated by experimental-active-percentile feature; default untouched.
//! Zero-trade -100 remains penalized, but excluded from normalization.
pub(super) fn blend(
    scores: &[f64],
    trades: &[f64],
    novelty: &[f64],
    novelty_weight: f64,
) -> Vec<f64> {
    assert_eq!(scores.len(), trades.len());
    assert_eq!(scores.len(), novelty.len());
    assert!(novelty_weight.is_finite() && (0.0..=1.0).contains(&novelty_weight));
    let mut eligible = (0..scores.len())
        .filter(|&i| scores[i].is_finite() && trades[i].is_finite() && trades[i] >= 1.0)
        .collect::<Vec<_>>();
    eligible.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then_with(|| a.cmp(&b)));
    let max_nov = novelty.iter().copied().filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max).max(1e-9);
    let mut out = vec![f64::NEG_INFINITY; scores.len()];
    let mut begin = 0;
    while begin < eligible.len() {
        let mut end = begin + 1;
        while end < eligible.len() && scores[eligible[begin]] == scores[eligible[end]] {
            end += 1;
        }
        let rank = if eligible.len() == 1 {
            0.5
        } else {
            (begin + end - 1) as f64 / (2 * (eligible.len() - 1)) as f64
        };
        for &index in &eligible[begin..end] {
            let nov = if novelty[index].is_finite() {
                (novelty[index] / max_nov).clamp(0.0, 1.0)
            } else { 0.0 };
            out[index] = (1.0 - novelty_weight) * rank + novelty_weight * nov;
        }
        begin = end;
    }
    for i in 0..out.len() {
        if scores[i].is_finite() && !(trades[i].is_finite() && trades[i] >= 1.0) {
            out[i] = -100.0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::blend;

    #[test]
    fn no_trade_sentinels_cannot_invert_real_fitness() {
        let fitness = [0.06, 0.01, -100., -100.];
        let trades = [60., 100., 0., 0.];
        let novelty = [0.1, 0.9, 1., 0.];
        let ranks = blend(&fitness,&trades,&novelty,0.2);
        assert!(ranks[0] > ranks[1], "{ranks:?}");
        assert_eq!(ranks[2],-100.);
        assert_eq!(ranks[3],-100.);
    }

    #[test]
    fn ties_are_stable_and_all_silent_fails_closed() {
        let scores = [0.2, 0.2, 0.2, -100.];
        let n = [10., 20., 40., 0.];
        let nov = [0.3, 0.3, 0.3, 0.9];
        let r = blend(&scores,&n,&nov,0.2);
        assert_eq!(r[0],r[1]);
        assert_eq!(r[1],r[2]);
        assert_eq!(r[2],-100.);
        assert_eq!(blend(&[-100.,-100.], &[0.,0.], &[0.8,1.0],0.2),vec![-100.,-100.]);
    }

    #[test]
    fn one_active_and_nonfinite_invalid_stay_bounded() {
        let r=blend(&[0.01, -100., f64::NEG_INFINITY], &[10.,0.,10.], &[0.1,1.,0.4],0.2);
        assert!(r[0].is_finite() && r[0] > r[1]);
        assert_eq!(r[1], -100.0);
        assert!(r[2].is_infinite() && r[2].is_sign_negative());
    }
}
