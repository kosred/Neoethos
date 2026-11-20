//! Closed-bar holding deadline for an owned live position. Broker closure and
//! ownership reconciliation remain the responsibility of the live loop.

pub(super) fn due(
    maximum_bars: usize,
    entry_bar: Option<i64>,
    timestamps: impl IntoIterator<Item = i64>,
) -> Result<bool, &'static str> {
    if maximum_bars == 0 {
        return Ok(false);
    }
    let entry = entry_bar
        .filter(|ts| *ts > 0)
        .ok_or("missing owned entry-bar timestamp")?;
    let mut previous = None;
    let mut saw_entry = false;
    let mut after_entry = 0usize;
    for timestamp in timestamps {
        if timestamp <= 0 || previous.is_some_and(|prior| timestamp <= prior) {
            return Err("holding history must contain positive, strictly increasing closed bars");
        }
        previous = Some(timestamp);
        saw_entry |= timestamp == entry;
        after_entry += usize::from(timestamp > entry);
    }
    if previous.is_none_or(|last| last < entry) {
        return Err("holding history does not reach the owned entry bar");
    }
    if after_entry >= maximum_bars {
        // Even a truncated tail proves at least this many closed bars after
        // the owned entry. No invented opening time or timeframe arithmetic.
        return Ok(true);
    }
    if !saw_entry {
        return Err("entry bar missing and retained history cannot prove the deadline");
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::due;

    #[test]
    fn disabled_policy_needs_no_position_history() {
        assert_eq!(due(0, None, []), Ok(false));
    }

    #[test]
    fn deadline_counts_bars_after_entry_and_includes_equality() {
        assert_eq!(due(2, Some(100), [90, 100, 110]), Ok(false));
        assert_eq!(due(2, Some(100), [90, 100, 110, 120]), Ok(true));
        assert_eq!(due(2, Some(100), [90, 100, 110, 120, 130]), Ok(true));
    }

    #[test]
    fn sparse_calendar_time_does_not_fabricate_extra_bars() {
        assert_eq!(due(2, Some(100), [100, 1_000_000]), Ok(false));
        assert_eq!(due(2, Some(100), [100, 1_000_000, 2_000_000]), Ok(true));
    }

    #[test]
    fn truncated_tail_can_prove_a_due_deadline_but_not_an_unexpired_one() {
        assert_eq!(due(2, Some(100), [120, 130]), Ok(true));
        assert!(due(3, Some(100), [120, 130]).is_err());
    }

    #[test]
    fn missing_or_future_entry_evidence_is_refused() {
        for entry in [None, Some(0), Some(-1), Some(140)] {
            assert!(due(2, entry, [100, 110, 120]).is_err());
        }
        assert!(due(2, Some(100), []).is_err());
    }

    #[test]
    fn unordered_duplicate_and_nonpositive_bars_are_refused() {
        for bars in [[100, 120, 110], [100, 110, 110], [0, 100, 110]] {
            assert!(due(2, Some(100), bars).is_err());
        }
    }

    #[test]
    fn a_new_owned_entry_does_not_inherit_the_previous_positions_age() {
        assert_eq!(due(2, Some(100), [100, 110, 120, 130]), Ok(true));
        assert_eq!(due(2, Some(130), [100, 110, 120, 130]), Ok(false));
    }
}
