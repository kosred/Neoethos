//! In-memory open-position tracking + bar-driven SL/TP management.
//!
//! Phase 1 is a self-contained simulator: positions are opened/closed/amended
//! from executed `TradeIntent`s and marked against replayed bars. P&L here is a
//! simple `points × volume` proxy — the authoritative strategy P&L stays in the
//! GA backtest; this only needs to prove the loop mechanics + exposure tracking.

use serde::{Deserialize, Serialize};

use crate::contracts::{
    CloseReason, Direction, ExecReport, ExecStatus, LiveBar, SignalSource, TradeIntent,
};

/// One open position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub id: String,
    pub symbol: String,
    pub dir: Direction,
    pub volume: f64,
    pub entry_price: f64,
    pub sl: Option<f64>,
    pub tp: Option<f64>,
    pub source: SignalSource,
    /// Engine bar index at which this position was opened. Drives the
    /// `max_hold_bars` time stop — the GA evaluator's third exit, which the
    /// replay did not have at all (audit #228). `serde(default)` so manifests
    /// written before 2026-08-09 still load.
    #[serde(default)]
    pub opened_at_bar: u64,
    /// The break-even / trailing stop level ratcheted in by PRIOR bars, or
    /// `None` while the trail has never armed (audit #227).
    ///
    /// It is a PRICE, not an offset, and it only ever moves in the favourable
    /// direction — exactly like `trail_px` in the GA evaluator
    /// (`neoethos-search/src/eval.rs:1566-1618`). `serde(default)` so manifests
    /// written before 2026-08-10 still load.
    #[serde(default)]
    pub trail_px: Option<f64>,
}

/// Break-even / trailing-stop geometry for the replay, in the SAME terms the GA
/// evaluator uses (audit #227).
///
/// Every field maps one-for-one onto `BacktestSettings`: `be_trigger_r` →
/// `trailing_be_trigger_r`, `stop_multiplier` → `trailing_atr_multiplier`
/// (which was never an ATR multiple — it is a multiple of the position's OWN
/// stop distance), `min_lock_pips` → `trailing_min_lock_pips`. `pip_size`
/// converts the lock from pips to price; the replay resolves it through
/// the exact broker `ProtoOASymbol` pip size, the same contract the bracket path
/// already uses, so no pip number is invented here.
///
/// The replay had NONE of this until 2026-08-10: it modelled the stop, the
/// target and the `max_hold_bars` time stop, while both the live loop
/// (`live_trading.rs:1479-1493`) and the evaluator also move the stop once a
/// trade reaches `+be_trigger_r × R`. A replay that does not model the exit a
/// strategy was scored under is not measuring that strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrailingPolicy {
    pub be_trigger_r: f64,
    pub stop_multiplier: f64,
    pub min_lock_pips: f64,
    pub pip_size: f64,
}

impl TrailingPolicy {
    /// `Some` only when every number can actually move a stop. A non-finite or
    /// non-positive trigger / multiplier / pip size, or a negative lock, would
    /// place the stop at a nonsense level, so those yield `None` and the caller
    /// reports the trail as UNARMED instead of running a trail nobody can read.
    pub fn new(
        be_trigger_r: f64,
        stop_multiplier: f64,
        min_lock_pips: f64,
        pip_size: f64,
    ) -> Option<Self> {
        let finite_positive = |v: f64| v.is_finite() && v > 0.0;
        if !finite_positive(be_trigger_r)
            || !finite_positive(stop_multiplier)
            || !finite_positive(pip_size)
            || !min_lock_pips.is_finite()
            || min_lock_pips < 0.0
        {
            return None;
        }
        Some(Self {
            be_trigger_r,
            stop_multiplier,
            min_lock_pips,
            pip_size,
        })
    }
}

impl Position {
    /// Unrealised P&L as `points × volume` (sign-aware). Pip value + contract
    /// size wire in with the real cost model later.
    pub fn unrealized(&self, price: f64) -> f64 {
        (price - self.entry_price) * self.dir.sign() * self.volume
    }

    /// The stop this bar is actually checked against: the position's own stop,
    /// or the trail ratcheted in by PRIOR bars when that is the tighter of the
    /// two (audit #227).
    ///
    /// The trail can only ever move the stop in the favourable direction, so
    /// this never widens risk — it is `max` for a long and `min` for a short,
    /// which is the same comparison the evaluator makes at
    /// `eval.rs:1571`/`:1597`.
    pub fn effective_sl(&self) -> Option<f64> {
        match (self.sl, self.trail_px) {
            (Some(sl), Some(trail)) if trail.is_finite() => Some(match self.dir {
                Direction::Long => sl.max(trail),
                Direction::Short => sl.min(trail),
                Direction::Flat => sl,
            }),
            (sl, _) => sl,
        }
    }

    /// Does `bar` cross this position's SL or TP? Returns the close reason + the
    /// price to close at (the level itself — a conservative fill assumption).
    /// SL is checked before TP so a bar that straddles both is treated as the
    /// adverse outcome (no intrabar-path optimism).
    ///
    /// The stop side uses [`Self::effective_sl`], i.e. the trail locked in by
    /// EARLIER bars only. This bar's own high must not move the stop its own low
    /// is checked against — that would be intra-bar look-ahead, and it is the
    /// ordering the evaluator is careful about in the same place.
    pub fn exit_hit(&self, bar: &LiveBar) -> Option<(CloseReason, f64)> {
        let sl = self.effective_sl();
        match self.dir {
            Direction::Long => {
                if let Some(sl) = sl {
                    if bar.l <= sl {
                        return Some((CloseReason::StopLoss, sl));
                    }
                }
                if let Some(tp) = self.tp {
                    if bar.h >= tp {
                        return Some((CloseReason::TakeProfit, tp));
                    }
                }
            }
            Direction::Short => {
                if let Some(sl) = sl {
                    if bar.h >= sl {
                        return Some((CloseReason::StopLoss, sl));
                    }
                }
                if let Some(tp) = self.tp {
                    if bar.l <= tp {
                        return Some((CloseReason::TakeProfit, tp));
                    }
                }
            }
            Direction::Flat => {}
        }
        None
    }

    /// Ratchet the trail from THIS bar's extreme, for the NEXT bar to be checked
    /// against. Returns `true` when the stored level moved.
    ///
    /// Runs only AFTER [`Self::exit_hit`] has declined to close the position, so
    /// the arithmetic is the evaluator's, in the evaluator's order
    /// (`eval.rs:1581-1592` long, `:1608-1618` short):
    ///
    /// * arm when the favourable excursion from entry reaches
    ///   `be_trigger_r × stop_distance`;
    /// * the candidate level is `extreme ∓ stop_multiplier × stop_distance`,
    ///   floored (long) / capped (short) at `entry ± min_lock_pips`;
    /// * keep it only if it is better than what is already locked in.
    ///
    /// `stop_distance` is the position's OWN risk, `|entry − sl|`. Without a
    /// stop there is no R, so a bracketless position never trails.
    pub fn ratchet_trail(&mut self, bar: &LiveBar, policy: &TrailingPolicy) -> bool {
        let Some(sl) = self.sl else {
            return false;
        };
        let stop_dist = (self.entry_price - sl).abs();
        if !stop_dist.is_finite() || stop_dist <= 0.0 {
            return false;
        }
        let trigger = policy.be_trigger_r * stop_dist;
        let lock = policy.min_lock_pips * policy.pip_size;
        let candidate = match self.dir {
            Direction::Long => {
                if (bar.h - self.entry_price) < trigger {
                    return false;
                }
                (bar.h - policy.stop_multiplier * stop_dist).max(self.entry_price + lock)
            }
            Direction::Short => {
                if (self.entry_price - bar.l) < trigger {
                    return false;
                }
                (bar.l + policy.stop_multiplier * stop_dist).min(self.entry_price - lock)
            }
            Direction::Flat => return false,
        };
        if !candidate.is_finite() {
            return false;
        }
        let better = match (self.trail_px, self.dir) {
            (None, _) => true,
            (Some(prev), Direction::Long) => candidate > prev,
            (Some(prev), Direction::Short) => candidate < prev,
            (Some(_), Direction::Flat) => false,
        };
        if better {
            self.trail_px = Some(candidate);
        }
        better
    }
}

/// Tracks open positions, applies executed intents, and emits SL/TP-driven
/// close intents per bar.
#[derive(Debug, Default)]
pub struct PositionManager {
    open: Vec<Position>,
    next_id: u64,
    realized_pnl: f64,
    opened_count: usize,
    closed_count: usize,
    /// Bar index of the bar currently being processed. Stamped onto every
    /// position opened on it.
    current_bar: u64,
    /// The GA evaluator's time stop. `None` ⇒ no time stop (the Phase-1
    /// behaviour); the real-gene replay paths set it from
    /// `EvaluationConfig::max_hold_bars` so a replayed trade cannot outlive
    /// every trade the backtest scored (audit #228).
    max_hold_bars: Option<u64>,
    /// Count of time-stop closes emitted, for the run report.
    max_hold_exits: usize,
    /// The GA evaluator's / live loop's break-even + trailing stop. `None` ⇒ the
    /// pre-2026-08-10 behaviour, in which the replay had no trail at all
    /// (audit #227) and a replayed trade could only end at its original stop,
    /// its target or the time stop.
    trailing: Option<TrailingPolicy>,
}

impl PositionManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tell the manager which bar index is being processed. The engine calls
    /// this once per bar, BEFORE managing exits and before applying fills.
    pub fn set_bar(&mut self, bar_index: u64) {
        self.current_bar = bar_index;
    }

    /// Arm the `max_hold_bars` time stop. `None` disables it.
    pub fn set_max_hold_bars(&mut self, bars: Option<u64>) {
        self.max_hold_bars = bars;
    }

    /// Arm the break-even / trailing stop. `None` disables it, which is what
    /// the replay did unconditionally before 2026-08-10.
    pub fn set_trailing(&mut self, policy: Option<TrailingPolicy>) {
        self.trailing = policy;
    }

    /// The armed trail, if any — so a caller can report the geometry it ran
    /// under instead of asserting it.
    pub fn trailing(&self) -> Option<TrailingPolicy> {
        self.trailing
    }

    /// How many positions were closed by the time stop.
    pub fn max_hold_exits(&self) -> usize {
        self.max_hold_exits
    }

    pub fn open_positions(&self) -> &[Position] {
        &self.open
    }

    /// Snapshot (clones) of the positions for one symbol — handed to the
    /// DecisionEngine so it can reason without holding a borrow on the manager.
    pub fn positions_for(&self, symbol: &str) -> Vec<Position> {
        self.open
            .iter()
            .filter(|p| p.symbol == symbol)
            .cloned()
            .collect()
    }

    pub fn has_open(&self, symbol: &str) -> bool {
        self.open.iter().any(|p| p.symbol == symbol)
    }

    pub fn open_count(&self) -> usize {
        self.open.len()
    }

    pub fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    pub fn opened_count(&self) -> usize {
        self.opened_count
    }

    pub fn closed_count(&self) -> usize {
        self.closed_count
    }

    /// Total unrealised P&L across all open positions marked at `mark`
    /// (a per-symbol price lookup; symbols with no mark contribute 0).
    pub fn unrealized_total(&self, mark: impl Fn(&str) -> Option<f64>) -> f64 {
        self.open
            .iter()
            .filter_map(|p| mark(&p.symbol).map(|px| p.unrealized(px)))
            .sum()
    }

    fn alloc_id(&mut self) -> String {
        self.next_id += 1;
        format!("sim-{}", self.next_id)
    }

    /// Reconcile an executed intent into the open set. No-op when the report is
    /// not `Filled` (a rejected/pending exec changes nothing on our books).
    pub fn apply(&mut self, intent: &TradeIntent, report: &ExecReport) {
        if report.status != ExecStatus::Filled {
            return;
        }
        match intent {
            TradeIntent::Open {
                symbol,
                dir,
                volume,
                sl,
                tp,
                source,
            } => {
                let price = report.fill_price.unwrap_or(0.0);
                let id = report
                    .position_id
                    .clone()
                    .unwrap_or_else(|| self.alloc_id());
                self.open.push(Position {
                    id,
                    symbol: symbol.clone(),
                    dir: *dir,
                    volume: *volume,
                    entry_price: price,
                    sl: *sl,
                    tp: *tp,
                    source: *source,
                    opened_at_bar: self.current_bar,
                    // Never armed at entry: the evaluator arms the trail only
                    // after a LATER bar's excursion reaches the trigger.
                    trail_px: None,
                });
                self.opened_count += 1;
            }
            TradeIntent::Close {
                position_id,
                volume,
                ..
            } => {
                if let Some(idx) = self.open.iter().position(|p| &p.id == position_id) {
                    let entry = self.open[idx].entry_price;
                    let sign = self.open[idx].dir.sign();
                    let fill = report.fill_price.unwrap_or(entry);
                    let pos_vol = self.open[idx].volume;
                    let close_vol = volume.unwrap_or(pos_vol).min(pos_vol);
                    self.realized_pnl += (fill - entry) * sign * close_vol;
                    if volume.is_none() || close_vol >= pos_vol {
                        self.open.remove(idx);
                        self.closed_count += 1;
                    } else {
                        self.open[idx].volume -= close_vol;
                    }
                }
            }
            TradeIntent::Amend {
                position_id,
                new_sl,
                new_tp,
            } => {
                if let Some(p) = self.open.iter_mut().find(|p| &p.id == position_id) {
                    if new_sl.is_some() {
                        p.sl = *new_sl;
                    }
                    if new_tp.is_some() {
                        p.tp = *new_tp;
                    }
                }
            }
            TradeIntent::Cancel { .. } => {}
        }
    }

    /// On each bar, produce `(Close intent, fill price)` pairs for every position
    /// of this symbol whose SL/TP the bar crossed — or whose `max_hold_bars`
    /// time stop has expired. The engine executes each at the returned level so
    /// realised P&L reflects the stop/target, not the bar close.
    ///
    /// ORDER OF PRECEDENCE: stop, then target, then the time stop. A bar that
    /// straddles both stop and target is already resolved adversely by
    /// [`Position::exit_hit`]; the time stop only fires on a bar where neither
    /// level was touched, and it fills at the bar CLOSE (the same price the GA
    /// evaluator uses for a `max_hold` exit).
    ///
    /// When a trail is armed ([`Self::set_trailing`], audit #227) the stop
    /// checked above is the one PRIOR bars ratcheted in, and this bar's own
    /// extreme then moves it for the next bar — the evaluator's order, not a
    /// bar that stops itself out on its own high.
    pub fn manage_on_bar(&mut self, bar: &LiveBar) -> Vec<(TradeIntent, f64)> {
        let max_hold = self.max_hold_bars;
        let current_bar = self.current_bar;
        let trailing = self.trailing;
        let mut out = Vec::new();
        let mut timed_out = 0usize;
        for p in self.open.iter_mut().filter(|p| p.symbol == bar.symbol) {
            if let Some((reason, price)) = p.exit_hit(bar) {
                out.push((
                    TradeIntent::Close {
                        position_id: p.id.clone(),
                        volume: None,
                        reason,
                    },
                    price,
                ));
                continue;
            }
            if let Some(limit) = max_hold {
                if current_bar.saturating_sub(p.opened_at_bar) >= limit {
                    timed_out += 1;
                    out.push((
                        TradeIntent::Close {
                            position_id: p.id.clone(),
                            volume: None,
                            reason: CloseReason::MaxHold,
                        },
                        bar.c,
                    ));
                    continue;
                }
            }
            // AFTER both exit checks, and only for a position that survived
            // this bar: ratchet the trail from this bar's extreme so the NEXT
            // bar is checked against it (audit #227). Doing it before the
            // checks would let a bar's own high move the stop its own low is
            // tested against.
            if let Some(policy) = trailing.as_ref() {
                p.ratchet_trail(bar, policy);
            }
        }
        self.max_hold_exits += timed_out;
        out
    }
}

#[cfg(test)]
mod trailing_tests {
    //! The replay's break-even / trailing stop, pinned against the GA
    //! evaluator's arithmetic (audit #227).
    //!
    //! These exist because the defect was an ABSENCE: the crate had no trail at
    //! all, so no test could fail. Each case states the level the evaluator
    //! produces for the same bar, so a future edit that changes the geometry has
    //! to change a written number rather than a behaviour nobody recorded.
    //!
    //! Fixture throughout: pip 0.0001, long entry 1.0000 with a 20-pip stop at
    //! 0.9980 (stop distance 0.0020), trigger 1.0R, multiplier 1.0, lock 2 pips
    //! — the production triple the 2026-08-09 measurement was taken under.

    use super::*;
    use crate::contracts::{ExecReport, ExecStatus};

    const PIP: f64 = 0.0001;

    fn policy() -> TrailingPolicy {
        TrailingPolicy::new(1.0, 1.0, 2.0, PIP).expect("the production triple must be usable")
    }

    fn bar(symbol: &str, h: f64, l: f64, c: f64) -> LiveBar {
        LiveBar {
            symbol: symbol.to_string(),
            tf: "M5".to_string(),
            o: c,
            h,
            l,
            c,
            volume: 1.0,
            ts: 0,
        }
    }

    fn opened_long(mgr: &mut PositionManager) {
        mgr.apply(
            &TradeIntent::Open {
                symbol: "EURUSD".to_string(),
                dir: Direction::Long,
                volume: 1.0,
                sl: Some(0.9980),
                tp: Some(1.0060),
                source: SignalSource::Strategy,
            },
            &ExecReport {
                status: ExecStatus::Filled,
                fill_price: Some(1.0000),
                position_id: Some("p1".to_string()),
                detail: String::new(),
            },
        );
    }

    fn long() -> Position {
        Position {
            id: "p1".to_string(),
            symbol: "EURUSD".to_string(),
            dir: Direction::Long,
            volume: 1.0,
            entry_price: 1.0000,
            sl: Some(0.9980),
            tp: Some(1.0060),
            source: SignalSource::Strategy,
            opened_at_bar: 0,
            trail_px: None,
        }
    }

    fn short() -> Position {
        Position {
            id: "p1".to_string(),
            symbol: "EURUSD".to_string(),
            dir: Direction::Short,
            volume: 1.0,
            entry_price: 1.0000,
            sl: Some(1.0020),
            tp: Some(0.9940),
            source: SignalSource::Strategy,
            opened_at_bar: 0,
            trail_px: None,
        }
    }

    #[test]
    fn the_trail_does_not_arm_below_the_trigger() {
        let mut p = long();
        // +10 pips of excursion against a 1.0R trigger on a 20-pip stop.
        assert!(!p.ratchet_trail(&bar("EURUSD", 1.0010, 0.9995, 1.0005), &policy()));
        assert_eq!(p.trail_px, None);
        assert_eq!(p.effective_sl(), Some(0.9980));
    }

    #[test]
    fn at_the_trigger_the_stop_moves_to_high_minus_one_r() {
        let mut p = long();
        // high 1.0025 = +12.5 pips >= 1.0R. candidate = 1.0025 - 0.0020 =
        // 1.0005; the 2-pip lock floor is 1.0002, so the candidate wins.
        assert!(p.ratchet_trail(&bar("EURUSD", 1.0025, 1.0000, 1.0020), &policy()));
        let trail = p.trail_px.expect("armed");
        assert!((trail - 1.0005).abs() < 1e-12, "trail {trail}");
        // Never widens: the effective stop is the tighter of the two.
        let eff = p.effective_sl().expect("stop");
        assert!((eff - 1.0005).abs() < 1e-12, "effective {eff}");
    }

    #[test]
    fn the_lock_floor_binds_when_the_bar_only_just_triggers() {
        let mut p = long();
        // high exactly 1.0R: candidate = 1.0020 - 0.0020 = 1.0000 (entry), so
        // the 2-pip minimum lock is what actually gets stored.
        assert!(p.ratchet_trail(&bar("EURUSD", 1.0020, 1.0000, 1.0015), &policy()));
        let trail = p.trail_px.expect("armed");
        assert!((trail - 1.0002).abs() < 1e-12, "trail {trail}");
    }

    #[test]
    fn the_trail_never_moves_backwards() {
        let mut p = long();
        p.ratchet_trail(&bar("EURUSD", 1.0040, 1.0010, 1.0035), &policy());
        let best = p.trail_px.expect("armed");
        // A later, lower high must not give the trade its risk back.
        assert!(!p.ratchet_trail(&bar("EURUSD", 1.0025, 1.0015, 1.0020), &policy()));
        assert_eq!(p.trail_px, Some(best));
    }

    #[test]
    fn a_bracketless_position_never_trails() {
        let mut p = long();
        p.sl = None;
        assert!(!p.ratchet_trail(&bar("EURUSD", 1.0100, 1.0000, 1.0090), &policy()));
        assert_eq!(p.trail_px, None);
    }

    #[test]
    fn the_short_side_mirrors_the_long_side() {
        let mut p = short();
        // low 0.9975 = -25 pips of excursion >= 1.0R on a 20-pip stop.
        // candidate = 0.9975 + 0.0020 = 0.9995; the lock cap is 0.9998, so the
        // candidate (the lower, tighter number) wins.
        assert!(p.ratchet_trail(&bar("EURUSD", 1.0000, 0.9975, 0.9980), &policy()));
        let trail = p.trail_px.expect("armed");
        assert!((trail - 0.9995).abs() < 1e-12, "trail {trail}");
        let eff = p.effective_sl().expect("stop");
        assert!((eff - 0.9995).abs() < 1e-12, "effective {eff}");
    }

    /// The ordering rule, which is the whole reason the ratchet runs LAST.
    #[test]
    fn a_bar_cannot_stop_itself_out_on_its_own_high() {
        let mut mgr = PositionManager::new();
        mgr.set_trailing(Some(policy()));
        opened_long(&mut mgr);
        // This bar arms the trail at 1.0005 from its own high — and its own low
        // of 1.0003 sits BELOW that level. It must not close: the level did not
        // exist while this bar was trading.
        let closes = mgr.manage_on_bar(&bar("EURUSD", 1.0025, 1.0003, 1.0020));
        assert!(closes.is_empty(), "closed on its own high: {closes:?}");
        assert_eq!(mgr.open_positions().len(), 1);
        // The NEXT bar is checked against it, and fills AT the level.
        let closes = mgr.manage_on_bar(&bar("EURUSD", 1.0010, 1.0000, 1.0002));
        assert_eq!(closes.len(), 1);
        let (_, fill) = &closes[0];
        assert!((fill - 1.0005).abs() < 1e-12, "fill {fill}");
    }

    /// `trailing: None` must be the pre-2026-08-10 engine, bar for bar.
    #[test]
    fn without_a_policy_the_stop_never_moves() {
        let mut mgr = PositionManager::new();
        mgr.set_trailing(None);
        opened_long(&mut mgr);
        assert!(
            mgr.manage_on_bar(&bar("EURUSD", 1.0025, 1.0000, 1.0020))
                .is_empty()
        );
        assert_eq!(mgr.open_positions()[0].trail_px, None);
        // A pullback to 1.0003 is nothing without a trail; only 0.9980 closes.
        assert!(
            mgr.manage_on_bar(&bar("EURUSD", 1.0010, 1.0003, 1.0005))
                .is_empty()
        );
        let closes = mgr.manage_on_bar(&bar("EURUSD", 1.0000, 0.9979, 0.9985));
        assert_eq!(closes.len(), 1);
        let (_, fill) = &closes[0];
        assert!((fill - 0.9980).abs() < 1e-12, "fill {fill}");
    }

    #[test]
    fn a_policy_that_cannot_move_a_stop_is_refused() {
        assert!(TrailingPolicy::new(f64::NAN, 1.0, 2.0, PIP).is_none());
        assert!(TrailingPolicy::new(1.0, 0.0, 2.0, PIP).is_none());
        assert!(TrailingPolicy::new(1.0, 1.0, -1.0, PIP).is_none());
        assert!(TrailingPolicy::new(1.0, 1.0, 2.0, 0.0).is_none());
        assert!(TrailingPolicy::new(1.0, 1.0, 0.0, PIP).is_some());
    }
}
