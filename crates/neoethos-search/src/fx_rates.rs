//! Quote -> account FX rates read from the exact canonical series selected for
//! the historical run.
//!
//! This module does not invent a rate. Cross-currency conversion remains
//! unavailable until the caller installs a verified [`CanonicalDatasetIdentity`]
//! anchor. Bridge bars must then come from the same external source namespace,
//! or from the same cTrader environment/server/account, and from a direct
//! canonical generation. Missing or ambiguous data produces no conversion.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, RwLock};

use neoethos_data::{CanonicalDatasetIdentity, CanonicalTimeframe};

use crate::data_selection::{CanonicalDataSelectionError, ExactCanonicalSeries};

/// Coarsest first: the bridge is needed only for its last close. Every member
/// is still a direct generation; the selector never derives one from M1.
const TIMEFRAME_PREFERENCE: [CanonicalTimeframe; 9] = [
    CanonicalTimeframe::D1,
    CanonicalTimeframe::H4,
    CanonicalTimeframe::H1,
    CanonicalTimeframe::M30,
    CanonicalTimeframe::M15,
    CanonicalTimeframe::M5,
    CanonicalTimeframe::M3,
    CanonicalTimeframe::M1,
    CanonicalTimeframe::W1,
];

fn store_selection() -> &'static RwLock<Option<ExactCanonicalSeries>> {
    static SELECTION: OnceLock<RwLock<Option<ExactCanonicalSeries>>> = OnceLock::new();
    SELECTION.get_or_init(|| RwLock::new(None))
}

fn cache() -> &'static Mutex<HashMap<(String, String), Option<f64>>> {
    static CACHE: OnceLock<Mutex<HashMap<(String, String), Option<f64>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Install the exact source/account series used by this historical run.
///
/// The anchor is verified before it becomes visible. Changing the root or the
/// anchor clears cached values so a later run cannot inherit a bridge from a
/// different source, cTrader server, or account.
pub fn set_store_selection(
    root: impl Into<std::path::PathBuf>,
    anchor: CanonicalDatasetIdentity,
) -> Result<(), CanonicalDataSelectionError> {
    let selection = ExactCanonicalSeries::open(root, anchor)?;
    let mut slot = store_selection().write().map_err(|error| {
        CanonicalDataSelectionError::InventoryFailed {
            requested_symbol: selection.anchor_identity().symbol_name().to_owned(),
            detail: format!("FX exact-selection lock is poisoned: {error}"),
        }
    })?;
    let unchanged = slot.as_ref().is_some_and(|current| {
        current.root() == selection.root()
            && current.anchor_identity() == selection.anchor_identity()
    });
    if unchanged {
        return Ok(());
    }
    *slot = Some(selection);
    if let Ok(mut values) = cache().lock() {
        values.clear();
    }
    Ok(())
}

/// How many units of `account` one unit of `quote` buys.
///
/// Matching currencies need no market data. Every cross-currency answer is
/// scoped by the installed exact anchor. The surrounding cost model already
/// treats `None` as an error; no numeric fallback is added here.
pub fn quote_to_account(quote: &str, account: &str) -> Option<f64> {
    let quote = quote.trim().to_ascii_uppercase();
    let account = account.trim().to_ascii_uppercase();
    if quote.is_empty() || account.is_empty() {
        return None;
    }
    if quote == account {
        return Some(1.0);
    }
    let key = (quote.clone(), account.clone());
    if let Ok(values) = cache().lock()
        && let Some(hit) = values.get(&key)
    {
        return *hit;
    }
    let resolved = resolve_uncached(&quote, &account);
    if let Ok(mut values) = cache().lock() {
        values.insert(key, resolved);
    }
    resolved
}

fn resolve_uncached(quote: &str, account: &str) -> Option<f64> {
    let selection = match store_selection().read() {
        Ok(slot) => slot.clone(),
        Err(error) => {
            tracing::error!(
                target: "neoethos_search::fx_rates",
                error = %error,
                "exact FX dataset-selection lock is poisoned"
            );
            return None;
        }
    }?;

    match last_close(&selection, &format!("{account}{quote}")) {
        Ok(price) => return Some(1.0 / price),
        Err(CanonicalDataSelectionError::MissingDirectTimeframe { .. }) => {}
        Err(error) => {
            tracing::error!(
                target: "neoethos_search::fx_rates",
                quote,
                account,
                anchor = %selection.anchor_identity().to_path_component(),
                error = %error,
                "exact inverse bridge selection failed; refusing another source/account"
            );
            return None;
        }
    }

    match last_close(&selection, &format!("{quote}{account}")) {
        Ok(price) => Some(price),
        Err(error) => {
            tracing::warn!(
                target: "neoethos_search::fx_rates",
                quote,
                account,
                anchor = %selection.anchor_identity().to_path_component(),
                error = %error,
                "no unambiguous direct canonical bridge exists in the selected source/account; \
                 no quote-to-account conversion was produced"
            );
            None
        }
    }
}

fn last_close(
    selection: &ExactCanonicalSeries,
    symbol: &str,
) -> Result<f64, CanonicalDataSelectionError> {
    let frame = selection.load_related_direct(symbol, &TIMEFRAME_PREFERENCE)?;
    frame
        .ohlcv()
        .close
        .last()
        .copied()
        .filter(|price| price.is_finite() && *price > 0.0)
        .ok_or_else(|| CanonicalDataSelectionError::ProvenanceMismatch {
            anchor_id: selection.anchor_identity().to_path_component(),
            detail: format!("direct canonical bridge {symbol} has no finite positive close"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_currencies_need_no_store() {
        assert_eq!(quote_to_account("GBP", "GBP"), Some(1.0));
        assert_eq!(quote_to_account("gbp", " GBP "), Some(1.0));
    }

    #[test]
    fn blank_currencies_resolve_to_nothing() {
        assert_eq!(quote_to_account("", "GBP"), None);
        assert_eq!(quote_to_account("USD", "  "), None);
    }

    #[test]
    fn no_exact_anchor_means_no_cross_currency_result() {
        if let Ok(mut slot) = store_selection().write() {
            *slot = None;
        }
        if let Ok(mut values) = cache().lock() {
            values.clear();
        }
        assert_eq!(quote_to_account("JPY", "GBP"), None);
    }
}
