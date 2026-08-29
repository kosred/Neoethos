mod app_services;

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};

use app_services::ctrader_data::{CTraderChartHistoryRequest, load_historical_bars_only};
use app_services::ctrader_live_auth::{
    CTraderEnvironment, CTraderLiveAuthBackend, CTraderTokenRefreshRequest,
    ProductionCTraderLiveAuthBackend,
};
use app_services::secure_store::production_ctrader_token_store;

const REFRESH_WINDOW_SECONDS: i64 = 120;

#[derive(Debug, PartialEq, Eq)]
struct ProbeArgs {
    symbol: String,
    timeframe: String,
    from_ms: i64,
    to_ms: i64,
    count: u32,
}

struct ProbeConnection {
    client_id: String,
    client_secret: String,
    access_token: String,
    environment: CTraderEnvironment,
    account_id: String,
}

fn build_request(args: &ProbeArgs, connection: ProbeConnection) -> CTraderChartHistoryRequest {
    CTraderChartHistoryRequest {
        client_id: connection.client_id,
        client_secret: connection.client_secret,
        access_token: connection.access_token,
        environment: connection.environment,
        account_id: connection.account_id,
        symbol_name: args.symbol.clone(),
        timeframe: args.timeframe.clone(),
        from_timestamp_ms: args.from_ms,
        to_timestamp_ms: args.to_ms,
        count: Some(args.count),
    }
}

fn now_unix_seconds() -> Result<i64> {
    i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
        .context("current Unix time exceeds i64")
}

fn parse_args(raw: impl IntoIterator<Item = String>) -> Result<ProbeArgs> {
    let mut symbol = None;
    let mut timeframe = None;
    let mut from_ms = None;
    let mut to_ms = None;
    let mut count = 10_u32;
    let mut args = raw.into_iter();
    while let Some(flag) = args.next() {
        let missing_value = || anyhow!("{flag} requires one value");
        match flag.as_str() {
            "--symbol" => symbol = Some(args.next().ok_or_else(missing_value)?),
            "--timeframe" => timeframe = Some(args.next().ok_or_else(missing_value)?),
            "--from-ms" => {
                from_ms = Some(
                    args.next()
                        .ok_or_else(missing_value)?
                        .parse::<i64>()
                        .context("--from-ms must be an i64 Unix millisecond")?,
                );
            }
            "--to-ms" => {
                to_ms = Some(
                    args.next()
                        .ok_or_else(missing_value)?
                        .parse::<i64>()
                        .context("--to-ms must be an i64 Unix millisecond")?,
                );
            }
            "--count" => {
                count = args
                    .next()
                    .ok_or_else(missing_value)?
                    .parse::<u32>()
                    .context("--count must be a positive u32")?;
            }
            _ => bail!("unknown argument {flag:?}"),
        }
    }

    let symbol = symbol.context("missing --symbol")?.trim().to_uppercase();
    let timeframe = timeframe
        .context("missing --timeframe")?
        .trim()
        .to_uppercase();
    let from_ms = from_ms.context("missing --from-ms")?;
    let to_ms = to_ms.context("missing --to-ms")?;
    if symbol.is_empty() || timeframe.is_empty() {
        bail!("--symbol and --timeframe must be non-empty");
    }
    if from_ms < 0 || to_ms <= from_ms {
        bail!("require 0 <= --from-ms < --to-ms");
    }
    if !(1..=100).contains(&count) {
        bail!("--count must be within 1..=100 for the bounded sample probe");
    }
    timeframe
        .parse::<neoethos_core::CanonicalTimeframe>()
        .map_err(|error| anyhow!("unsupported direct broker timeframe {timeframe}: {error}"))?;

    Ok(ProbeArgs {
        symbol,
        timeframe,
        from_ms,
        to_ms,
        count,
    })
}

fn main() -> Result<()> {
    let args = parse_args(std::env::args().skip(1))?;
    let credentials_path = neoethos_core::broker_config::credentials_file_path()?;
    let settings = neoethos_core::broker_config::load_from_disk(&credentials_path)?
        .with_context(|| format!("no broker credentials at {}", credentials_path.display()))?;
    let ctrader = settings.ctrader;
    if ctrader.client_id.trim().is_empty() || ctrader.client_secret.trim().is_empty() {
        bail!("canonical broker credentials are missing client id/secret");
    }
    let account = ctrader
        .accounts
        .iter()
        .find(|account| account.enabled_for_execution)
        .or_else(|| ctrader.accounts.first())
        .context("canonical broker credentials contain no account")?;
    account
        .account_id
        .parse::<i64>()
        .context("configured cTrader account id is not numeric")?;

    let token_store = production_ctrader_token_store();
    let mut token = token_store
        .load_token_bundle_with_legacy_fallback()?
        .context("no cTrader OAuth token in the OS credential store")?;
    if token.needs_refresh_at(now_unix_seconds()?, REFRESH_WINDOW_SECONDS) {
        token =
            ProductionCTraderLiveAuthBackend.refresh_token_bundle(&CTraderTokenRefreshRequest {
                client_id: ctrader.client_id.clone(),
                client_secret: ctrader.client_secret.clone(),
                refresh_token: token.refresh_token.clone(),
                scope: token.scope.clone(),
            })?;
        token_store.save_token_bundle(&token)?;
    }

    let environment = match ctrader.environment {
        neoethos_core::broker_config::CTraderBrokerEnvironment::Live => CTraderEnvironment::Live,
        neoethos_core::broker_config::CTraderBrokerEnvironment::Demo => CTraderEnvironment::Demo,
    };
    let expected_timeframe = args
        .timeframe
        .parse::<neoethos_core::CanonicalTimeframe>()?;
    let result = load_historical_bars_only(&build_request(
        &args,
        ProbeConnection {
            client_id: ctrader.client_id,
            client_secret: ctrader.client_secret,
            access_token: token.access_token,
            environment,
            account_id: account.account_id.clone(),
        },
    ))?;
    result.validate_identity(result.symbol_id, expected_timeframe)?;
    if !result.symbol.symbol_name.eq_ignore_ascii_case(&args.symbol) {
        bail!(
            "broker returned symbol {:?} for requested {:?}",
            result.symbol.symbol_name,
            args.symbol
        );
    }
    if result.bars.is_empty() {
        bail!(
            "broker returned no direct {} {} sample bars for [{}, {}]",
            args.symbol,
            args.timeframe,
            args.from_ms,
            args.to_ms
        );
    }

    println!(
        "sample symbol={} symbol_id={} timeframe={} bars={} has_more={}",
        result.symbol.symbol_name,
        result.symbol_id,
        result.timeframe,
        result.bars.len(),
        result.has_more
    );
    for (index, bar) in result.bars.iter().enumerate() {
        println!(
            "bar[{index}] timestamp_ms={} open={:.10} high={:.10} low={:.10} close={:.10} volume={:?}",
            bar.timestamp_ms, bar.open, bar.high, bar.low, bar.close, bar.volume
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::app_services::ctrader_live_auth::CTraderEnvironment;
    use super::{ProbeArgs, ProbeConnection, build_request, parse_args};

    fn required_args() -> Vec<String> {
        [
            "--symbol",
            "eurusd",
            "--timeframe",
            "m5",
            "--from-ms",
            "1000",
            "--to-ms",
            "2000",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn parser_accepts_only_a_bounded_direct_broker_sample() {
        let parsed = parse_args(required_args()).expect("valid bounded direct sample");
        assert_eq!(
            parsed,
            ProbeArgs {
                symbol: "EURUSD".to_owned(),
                timeframe: "M5".to_owned(),
                from_ms: 1000,
                to_ms: 2000,
                count: 10,
            }
        );

        let mut zero_count = required_args();
        zero_count.extend(["--count".to_owned(), "0".to_owned()]);
        assert!(
            parse_args(zero_count)
                .unwrap_err()
                .to_string()
                .contains("1..=100")
        );

        let mut oversized = required_args();
        oversized.extend(["--count".to_owned(), "101".to_owned()]);
        assert!(
            parse_args(oversized)
                .unwrap_err()
                .to_string()
                .contains("1..=100")
        );
    }

    #[test]
    fn parser_accepts_every_direct_ctrader_timeframe_and_rejects_synthetic_ones() {
        for timeframe in [
            "M1", "M2", "M3", "M4", "M5", "M10", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1",
        ] {
            let mut args = required_args();
            args[3] = timeframe.to_owned();
            assert_eq!(
                parse_args(args)
                    .expect("real cTrader timeframe must be accepted")
                    .timeframe,
                timeframe
            );
        }

        for timeframe in ["S1", "M6", "M20", "H2", "D2"] {
            let mut args = required_args();
            args[3] = timeframe.to_owned();
            assert!(
                parse_args(args)
                    .unwrap_err()
                    .to_string()
                    .contains("unsupported direct broker timeframe"),
                "{timeframe} must not enter the direct-broker probe"
            );
        }
    }

    #[test]
    fn parser_is_strict_about_window_and_unknown_arguments() {
        let mut reversed = required_args();
        reversed[5] = "2000".to_owned();
        reversed[7] = "1000".to_owned();
        assert!(
            parse_args(reversed)
                .unwrap_err()
                .to_string()
                .contains("0 <= --from-ms")
        );

        let mut unknown = required_args();
        unknown.push("--resample".to_owned());
        assert!(
            parse_args(unknown)
                .unwrap_err()
                .to_string()
                .contains("unknown argument")
        );
    }

    #[test]
    fn request_preserves_the_exact_direct_window_without_resampling() {
        let args = parse_args(required_args()).expect("valid bounded direct sample");
        let request = build_request(
            &args,
            ProbeConnection {
                client_id: "client".to_owned(),
                client_secret: "secret".to_owned(),
                access_token: "token".to_owned(),
                environment: CTraderEnvironment::Demo,
                account_id: "123".to_owned(),
            },
        );

        assert_eq!(request.symbol_name, "EURUSD");
        assert_eq!(request.timeframe, "M5");
        assert_eq!(request.from_timestamp_ms, 1000);
        assert_eq!(request.to_timestamp_ms, 2000);
        assert_eq!(request.count, Some(10));
        assert_eq!(request.environment, CTraderEnvironment::Demo);
        assert_eq!(request.account_id, "123");
    }

    #[test]
    fn production_probe_reads_real_broker_bars_without_publish_or_synthesis() {
        let source = include_str!("main.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("production source prefix");
        for required in [
            "production_ctrader_token_store",
            "load_historical_bars_only",
            "result.validate_identity",
            "bar.volume",
            "fn main() -> Result<()>",
        ] {
            assert!(production.contains(required), "missing {required}");
        }
        for forbidden in [
            "resample_timeframe",
            "synthesize_timeframe",
            "CanonicalOhlcvWriter",
            "std::fs::",
        ] {
            assert!(!production.contains(forbidden), "forbidden {forbidden}");
        }
    }
}
