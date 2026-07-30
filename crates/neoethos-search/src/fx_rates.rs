//! quote → account FX rates for the backtest, read from the local store.
//!
//! P&L is money, and money has a currency. A EURJPY pip is worth 1 000 JPY per
//! lot; on a GBP account that is about £5. Without the conversion the backtest
//! books it as £1 000 — every JPY result inflated roughly 192-fold, every
//! USD-quoted result by about 27 %. Since the search ranks candidates on
//! profit, the inflated ones win selection and reach the live account, where
//! they earn what they were always worth. That is a mechanical backtest-to-live
//! divergence, not a strategy failure.
//!
//! Live trading already resolves this rate from the broker
//! (`live_trading::resolve_quote_to_account_rate`), which is why live sizing was
//! right while the backtest was wrong. Discovery runs offline, so it takes the
//! same approach against the data it is already reading: the bridging pair, from
//! the store.
//!
//! The root has to be installed by the caller because there is no global
//! `Settings` in this process. When it is missing, or the bridging pair is not
//! in the store, this returns `None` — and the cost model treats that as an
//! error rather than substituting a number, because a wrong pip value is not
//! visible in any result it produces.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock, RwLock};

/// Coarsest first: the bridging pair is only needed for its last close, and a
/// D1 series is a fraction of the bars of an M1 one.
const TIMEFRAME_PREFERENCE: [&str; 9] = ["D1", "H4", "H1", "M30", "M15", "M5", "M3", "M1", "W1"];

fn store_root() -> &'static RwLock<Option<PathBuf>> {
    static ROOT: OnceLock<RwLock<Option<PathBuf>>> = OnceLock::new();
    ROOT.get_or_init(|| RwLock::new(None))
}

fn cache() -> &'static Mutex<HashMap<(String, String), Option<f64>>> {
    static CACHE: OnceLock<Mutex<HashMap<(String, String), Option<f64>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Point the resolver at the data store. Call once, before discovery.
///
/// Changing the root clears the cache, so a run against a different store
/// cannot inherit the previous one's rates.
pub fn set_store_root(root: impl Into<PathBuf>) {
    let root = root.into();
    if let Ok(mut slot) = store_root().write() {
        if slot.as_ref() == Some(&root) {
            return;
        }
        *slot = Some(root);
    }
    if let Ok(mut cache) = cache().lock() {
        cache.clear();
    }
}

/// The installed store root, if any.
pub fn configured_store_root() -> Option<PathBuf> {
    store_root().read().ok().and_then(|slot| slot.clone())
}

/// How many units of `account` one unit of `quote` buys.
///
/// `Some(1.0)` when the currencies match. Otherwise the bridging pair is read
/// from the store in whichever orientation exists: `{ACCOUNT}{QUOTE}` quotes
/// QUOTE per ACCOUNT so the rate is its reciprocal, `{QUOTE}{ACCOUNT}` quotes it
/// directly.
///
/// Cached per pair, including negative results — a missing bridging pair will
/// not start existing mid-run, and re-reading a vortex file per candidate would
/// cost more than the search.
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
    if let Ok(cache) = cache().lock()
        && let Some(hit) = cache.get(&key)
    {
        return *hit;
    }
    let resolved = resolve_uncached(&quote, &account);
    if let Ok(mut cache) = cache().lock() {
        cache.insert(key, resolved);
    }
    resolved
}

fn resolve_uncached(quote: &str, account: &str) -> Option<f64> {
    let root = configured_store_root()?;
    if let Some(price) = last_close(&root, &format!("{account}{quote}"))
        && price > 0.0
    {
        return Some(1.0 / price);
    }
    if let Some(price) = last_close(&root, &format!("{quote}{account}")) {
        return Some(price);
    }
    tracing::warn!(
        target: "neoethos_search::fx_rates",
        quote,
        account,
        "no bridging pair in the store for the quote→account conversion; \
         download {account}{quote} or {quote}{account}, or set \
         models.eval_runtime.quote_to_account_rate"
    );
    None
}

fn last_close(root: &PathBuf, symbol: &str) -> Option<f64> {
    let timeframes = neoethos_data::discover_timeframes(root, symbol).ok()?;
    if timeframes.is_empty() {
        return None;
    }
    let chosen = TIMEFRAME_PREFERENCE
        .iter()
        .find(|tf| timeframes.iter().any(|have| have.eq_ignore_ascii_case(tf)))
        .map(|tf| (*tf).to_string())
        .or_else(|| timeframes.first().cloned())?;
    let ohlcv = neoethos_data::load_symbol_timeframe_tail(root, symbol, &chosen, 1).ok()?;
    ohlcv
        .close
        .last()
        .copied()
        .filter(|price| price.is_finite() && *price > 0.0)
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

    /// Without a root there is no data to read, and the answer must be "unknown"
    /// rather than a plausible-looking number — the whole point of the module.
    #[test]
    fn an_unresolvable_pair_yields_none_not_a_guess() {
        assert_eq!(quote_to_account("JPY", "GBP"), None);
    }

    /// Against the operator's real store, because a resolver that works on
    /// synthetic fixtures and not on the actual layout would have fixed nothing.
    /// Skips where that store is absent rather than failing, so this stays
    /// runnable on a machine that has never downloaded data.
    #[test]
    fn the_real_store_resolves_the_bridges_this_account_needs() {
        let Some(local) = std::env::var_os("LOCALAPPDATA") else {
            return;
        };
        let root = PathBuf::from(local).join("neoethos").join("data");
        if !root.join("symbol=GBPUSD").exists() {
            return;
        }
        set_store_root(&root);

        // USD→GBP via GBPUSD: the price is USD per GBP, so the rate is its
        // reciprocal and a $10 pip becomes roughly £8.
        let usd = quote_to_account("USD", "GBP").expect("GBPUSD is in the store");
        assert!(
            (0.6..0.95).contains(&usd),
            "USD→GBP resolved to {usd}, which is not a plausible rate"
        );

        // JPY→GBP via GBPJPY — the conversion that was missing entirely, turning
        // a 1 000 JPY pip into about £5 instead of £1 000.
        if root.join("symbol=GBPJPY").exists() {
            let jpy = quote_to_account("JPY", "GBP").expect("GBPJPY is in the store");
            assert!(
                (0.003..0.01).contains(&jpy),
                "JPY→GBP resolved to {jpy}, which is not a plausible rate"
            );
            let pip_value = 1000.0 * jpy;
            assert!(
                (3.0..10.0).contains(&pip_value),
                "a EURJPY pip came out at £{pip_value} per lot; the defect booked £1000"
            );
        }
    }
}
