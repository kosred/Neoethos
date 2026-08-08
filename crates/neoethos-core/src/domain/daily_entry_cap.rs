//! Account-wide per-UTC-day entry counter — the enforcement behind
//! `risk.max_trades_per_day`.
//!
//! # Why this exists
//!
//! The operator config has carried `risk.max_trades_per_day: 8` for months and
//! NOTHING on the live entry path ever read it. Measured on the real journal
//! (238 closed trades) the engines took 20–47 entries per day EACH; a cap of 8
//! would have refused 68.1 % of them, over a period where costs ran 152 % of
//! gross profit. This module gives the key a recipient.
//!
//! # Counter scope: the ACCOUNT, not the engine
//!
//! Live engines are per-portfolio and several run at once against the SAME
//! broker account. A per-engine counter would quietly turn "8" into
//! `8 × engines` — 32 account-wide with four engines running — which is not
//! what an operator reading "8" expects. So there is ONE counter, shared by
//! every engine in the process (`live_trading.rs` holds it in a `static`), and
//! the cap binds on the account's total. An earlier draft of this rule
//! (commit 715058fe, never merged) counted per engine; that scope was its
//! known weakness and is deliberately not reproduced here.
//!
//! # Honest limits
//!
//! * The counter lives in process memory: it resets at UTC midnight AND on app
//!   restart. Entries opened by a previous app run on the same day are not
//!   counted. The journal (closed trades) cannot seed it either — a trade
//!   opened today and still open is in no journal yet.
//! * Two engines can hold the lock only one at a time, so the reserve/release
//!   protocol cannot over-admit: a slot is taken BEFORE the order goes out and
//!   given back if the order never becomes an entry.
//!
//! # Default OFF
//!
//! Enforcement is armed only by `risk.max_trades_per_day_enabled: true`
//! (default `false`). Disarmed, [`AccountDailyEntryCap::try_reserve`] still
//! counts — so the day's number is accurate in logs — but never refuses:
//! today's behaviour is unchanged until the operator flips the key.

use std::sync::Mutex;

/// Why an entry was refused: the account already used `count` of `cap`
/// entries this UTC day. Carried whole so the caller can log the numbers —
/// a refusal the operator cannot explain is a control he will disable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DailyEntryCapRefusal {
    /// Entries already opened on the account this UTC day.
    pub count: u32,
    /// The configured `risk.max_trades_per_day`.
    pub cap: u32,
}

/// One shared per-account counter of live entries per UTC day.
///
/// `utc_day` is an opaque day key (the caller uses `yyyymmdd` — the IDENTICAL
/// key the inline daily-loss breaker computes, so the two rules can never
/// disagree about which day an entry belongs to). A new key resets the count.
pub struct AccountDailyEntryCap {
    /// (utc day key, entries reserved for that day).
    state: Mutex<(u32, u32)>,
}

impl AccountDailyEntryCap {
    /// `const` so `live_trading.rs` can hold the account-wide instance in a
    /// plain `static` — one counter per process, shared by every engine.
    pub const fn new() -> Self {
        Self {
            state: Mutex::new((0, 0)),
        }
    }

    /// Reserve one entry slot for `utc_day`.
    ///
    /// * `cap = Some(n)` — armed: refuses (without counting) once `n` slots
    ///   are taken this day, returning the numbers for the refusal log.
    /// * `cap = None` — disarmed: always admits, but still counts, so the
    ///   day total stays accurate for logging/status.
    ///
    /// On `Ok(k)`, `k` is the day's entry number (1-based). The slot is taken
    /// IMMEDIATELY — before the order is submitted — so two engines racing at
    /// `count = cap − 1` cannot both pass. If the order then fails to become
    /// an entry, give the slot back with [`release`](Self::release).
    pub fn try_reserve(
        &self,
        utc_day: u32,
        cap: Option<u32>,
    ) -> Result<u32, DailyEntryCapRefusal> {
        // A poisoned lock must not kill the trading loop (a panic while
        // holding it cannot leave the two u32s half-written): take the data.
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if s.0 != utc_day {
            *s = (utc_day, 0);
        }
        if let Some(cap) = cap
            && s.1 >= cap
        {
            return Err(DailyEntryCapRefusal { count: s.1, cap });
        }
        s.1 += 1;
        Ok(s.1)
    }

    /// Give back a slot reserved by [`try_reserve`](Self::try_reserve) whose
    /// order never became an entry (placement failed, or a later gate skipped
    /// the bar). A no-op if the day has already rolled — yesterday's count no
    /// longer binds anything.
    pub fn release(&self, utc_day: u32) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if s.0 == utc_day && s.1 > 0 {
            s.1 -= 1;
        }
    }

    /// Entries counted for `utc_day` (0 if the day has rolled).
    pub fn count_for(&self, utc_day: u32) -> u32 {
        let s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if s.0 == utc_day { s.1 } else { 0 }
    }
}

impl Default for AccountDailyEntryCap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: u32 = 2026_08_08;

    /// The operative promise: armed with cap 8, the 9th entry of the same UTC
    /// day is refused — and the refusal carries the numbers for the log.
    #[test]
    fn ninth_entry_same_utc_day_refused_when_armed_with_cap_8() {
        let cap = AccountDailyEntryCap::new();
        for k in 1..=8 {
            assert_eq!(cap.try_reserve(DAY, Some(8)), Ok(k));
        }
        assert_eq!(
            cap.try_reserve(DAY, Some(8)),
            Err(DailyEntryCapRefusal { count: 8, cap: 8 })
        );
        // The refusal itself must not have consumed a slot.
        assert_eq!(cap.count_for(DAY), 8);
    }

    /// Default-OFF promise: with the flag off (cap = None) the 9th — and any
    /// later — entry is admitted; today's behaviour is unchanged.
    #[test]
    fn ninth_entry_allowed_when_flag_off() {
        let cap = AccountDailyEntryCap::new();
        for k in 1..=9 {
            assert_eq!(cap.try_reserve(DAY, None), Ok(k));
        }
        // Still counted accurately even while disarmed.
        assert_eq!(cap.count_for(DAY), 9);
    }

    /// The cap is per UTC DAY: a new day key resets the count to zero.
    #[test]
    fn utc_day_roll_resets_the_count() {
        let cap = AccountDailyEntryCap::new();
        for _ in 0..8 {
            cap.try_reserve(DAY, Some(8)).unwrap();
        }
        assert!(cap.try_reserve(DAY, Some(8)).is_err());
        assert_eq!(cap.try_reserve(DAY + 1, Some(8)), Ok(1));
        assert_eq!(cap.count_for(DAY), 0);
    }

    /// A failed order gives its slot back — the cap counts ENTRIES, not
    /// attempts.
    #[test]
    fn release_returns_the_slot() {
        let cap = AccountDailyEntryCap::new();
        for _ in 0..8 {
            cap.try_reserve(DAY, Some(8)).unwrap();
        }
        assert!(cap.try_reserve(DAY, Some(8)).is_err());
        cap.release(DAY);
        assert_eq!(cap.try_reserve(DAY, Some(8)), Ok(8));
    }

    /// Releasing after the day rolled must not corrupt the new day's count.
    #[test]
    fn release_after_day_roll_is_a_noop() {
        let cap = AccountDailyEntryCap::new();
        cap.try_reserve(DAY, Some(8)).unwrap();
        assert_eq!(cap.try_reserve(DAY + 1, Some(8)), Ok(1));
        cap.release(DAY); // stale release for yesterday
        assert_eq!(cap.count_for(DAY + 1), 1);
    }
}
