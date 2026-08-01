//! Backend-independent contracts for the NeoEthos GPU-native discovery pipeline.
//!
//! This crate has no GPU runtime dependency. It separates ergonomic host DTOs
//! from stable C-compatible POD layouts shared by CubeCL, native CUDA and Rust
//! FFI callers.

use serde::{Deserialize, Serialize};

pub const ABI_VERSION: u32 = 3;
pub const POPULATION_SETTINGS_FLAG_RISK_BASED_SIZING: u32 = 1 << 0;
pub const POPULATION_PRECEDENCE_STOP_FIRST: u32 = 0;
pub const POPULATION_DIRECTION_LONG: i32 = 1;
pub const POPULATION_DIRECTION_SHORT: i32 = -1;
pub const POPULATION_EXIT_NONE: i32 = 0;
pub const POPULATION_EXIT_STOP: i32 = 1;
pub const POPULATION_EXIT_TARGET: i32 = 2;
pub const POPULATION_EXIT_MAX_HOLD: i32 = 3;
pub const POPULATION_EXIT_GAP: i32 = 4;

pub mod host {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct DatasetDto {
        pub timestamps: Vec<i64>,
        pub open: Vec<f64>,
        pub high: Vec<f64>,
        pub low: Vec<f64>,
        pub close: Vec<f64>,
        pub feature_names: Vec<String>,
        /// Feature-major contiguous values: `[feature][row]`.
        pub features: Vec<f32>,
        pub months: Vec<i64>,
        pub days: Vec<i64>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct GeneBatchDto {
        pub candidate_ids: Vec<u64>,
        pub offsets: Vec<u32>,
        pub indices: Vec<u32>,
        pub weights: Vec<f32>,
        pub long_thresholds: Vec<f32>,
        pub short_thresholds: Vec<f32>,
        pub stop_ticks: Vec<i64>,
        pub target_ticks: Vec<i64>,
        pub stop_vol_multipliers: Vec<f32>,
        pub flags: Vec<u32>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ScenarioBatchDto {
        pub scenarios: Vec<super::device::ScenarioDescriptor>,
        pub perturbations: Vec<f32>,
        pub absolute_indices: Vec<u32>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CompactSurvivorDto {
        pub candidate_id: u64,
        pub rank_fields: Vec<i64>,
        pub metrics: super::device::Metrics,
    }
}

pub mod device {
    use super::*;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct BufferRef {
        pub offset: u64,
        pub len: u64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct HandleToken {
        pub session_id: u64,
        pub backend_id: u32,
        pub device_id: u32,
        pub generation: u64,
        pub buffer_kind: u32,
        pub reserved: u32,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
    pub struct DatasetHeader {
        pub abi_version: u32,
        pub flags: u32,
        pub row_count: u64,
        pub feature_count: u32,
        pub price_scale_exp: i32,
        pub timestamps: BufferRef,
        pub open: BufferRef,
        pub high: BufferRef,
        pub low: BufferRef,
        pub close: BufferRef,
        pub features: BufferRef,
        pub months: BufferRef,
        pub days: BufferRef,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
    pub struct GeneDescriptor {
        pub candidate_id: u64,
        pub term_offset: u32,
        pub term_count: u32,
        pub long_threshold: f32,
        pub short_threshold: f32,
        pub stop_ticks: i64,
        pub target_ticks: i64,
        pub stop_vol_multiplier: f32,
        pub flags: u32,
        pub reserved: u64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ScenarioDescriptor {
        pub base_candidate_id: u64,
        pub scenario_id: u64,
        pub rng_counter: u64,
        pub window_offset: u64,
        pub window_len: u32,
        pub scenario_type: u32,
        pub spread_ticks: i32,
        pub slippage_ticks: i32,
        pub commission_micros: i64,
        pub perturbation_offset: u32,
        pub perturbation_count: u32,
        pub reserved: u64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
    pub struct TradeOutcome {
        pub candidate_id: u64,
        pub scenario_id: u64,
        pub entry_bar: u32,
        pub exit_bar: u32,
        pub exit_reason: u32,
        pub direction: i32,
        pub pnl_micros: i64,
        pub equity_after_micros: i64,
        pub reserved: u64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
    pub struct Metrics {
        pub candidate_id: u64,
        pub scenario_id: u64,
        pub net_profit: f64,
        pub max_drawdown: f64,
        pub sharpe: f64,
        pub profit_factor: f64,
        pub win_rate: f64,
        pub trade_count: u64,
        pub monthly_target_hit_rate: f64,
        pub flags: u32,
        pub reserved: u32,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct PropFirmState {
        pub equity_micros: i64,
        pub peak_equity_micros: i64,
        pub day_start_equity_micros: i64,
        pub month_start_equity_micros: i64,
        pub current_day_id: i64,
        pub current_month_id: i64,
        pub trading_days: u32,
        pub flags: u32,
    }

    /// Fixed-width settings shared by Prototype B/C native entry points.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
    pub struct NeoPopulationSettings {
        pub abi_version: u32,
        pub flags: u32,
        pub max_hold_bars: u32,
        pub min_hold_bars: u32,
        pub max_trades_per_day: u32,
        pub month_capacity: u32,
        pub gap_threshold_ms: i64,
        pub initial_equity: f64,
        pub pip_value: f64,
        pub spread_pips: f64,
        pub commission_per_trade: f64,
        pub pip_value_per_lot: f64,
        pub swap_long_pips_per_day: f64,
        pub swap_short_pips_per_day: f64,
        pub pnl_conversion_fee_rate: f64,
        pub risk_per_trade_min: f64,
        pub risk_per_trade_max: f64,
        pub high_quality_confidence: f64,
        pub adaptive_rr: f64,
        /// Trailing stop. The kernel had none while the CPU engine applied one,
        /// so any run with trailing on was comparing two different strategies.
        ///
        /// Kept as its own `u32` rather than a bit in `flags` so the four values
        /// travel together and a run cannot half-configure the feature.
        pub trailing_enabled: u32,
        pub _trailing_pad: u32,
        /// How far behind the best price seen the trail sits, as a fraction of
        /// the position's own stop distance.
        pub trailing_atr_multiplier: f64,
        /// Profit, in units of the stop distance, before the trail arms at all.
        pub trailing_be_trigger_r: f64,
        /// Floor for the trail, in pips above entry, so what is protected does
        /// not vary per gene and fall below the cost of the trade.
        pub trailing_min_lock_pips: f64,
        /// Spread in pips per liquidity window, resolved from the entry bar's
        /// UTC hour exactly as `BacktestSettings::spread_pips_for_bar` does.
        ///
        /// The scalar `spread_pips` charged one number at every hour. Real
        /// spread is not one number: the London/NY overlap is typically 30-50 %
        /// of the Asian session's, so a flat charge over-pays the liquid hours
        /// and under-pays the thin ones — and a search that under-pays the thin
        /// hours will prefer strategies that trade in them. The CPU has
        /// resolved per bar since the profile type existed; the device had no
        /// field to receive it, so enabling it would have made the two lanes
        /// disagree with nothing to catch it.
        ///
        /// When no profile is configured the host writes `spread_pips` into all
        /// three, so the bucket lookup returns the scalar and the result is
        /// bit-identical to before. There is no flag and no branch to get wrong.
        pub spread_pips_asian: f64,
        pub spread_pips_overlap: f64,
        pub spread_pips_late_ny: f64,
    }

    /// Canonical causal entry emitted from a signal observed on the prior bar.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
    pub struct NeoPopulationEvent {
        pub candidate_id: u64,
        pub scenario_id: u64,
        pub entry_bar: u32,
        pub last_bar: u32,
        pub direction: i32,
        pub precedence: u32,
        pub stop_price: f64,
        pub target_price: f64,
        /// Fill price of the entry, spread already applied.
        ///
        /// The reducer recomputes this locally, but the first-hit walk needs it
        /// too: excursion is measured against the entry, and that walk is the
        /// only place that sees every bar the position was open.
        pub entry_price: f64,
    }

    /// First-hit result positionally aligned with its input event.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct NeoPopulationOutcome {
        pub candidate_id: u64,
        pub scenario_id: u64,
        pub exit_bar: i32,
        pub exit_reason: i32,
        /// Bar the position was opened on, so a host can rebuild the trade's
        /// span without re-deriving it from the event stream.
        pub entry_bar: i32,
        pub _pad: i32,
        /// Max favourable excursion, in account currency, per lot.
        ///
        /// Mirrors the CPU walk exactly: updated on every bar the position is
        /// open — including the exit bar, and before any exit test — as
        /// `max(0, (high - entry) / pip * pip_value_per_lot)` for a long and
        /// the mirrored form for a short. Not scaled by position size.
        pub mfe: f64,
        /// Max adverse excursion, same convention, reported positive.
        pub mae: f64,
        /// Price the position actually closed at.
        ///
        /// The reducer used to rebuild this from the exit reason —
        /// `event.stop_price` for a stop, `event.target_price` for a target.
        /// That works only while the levels are fixed at entry. A trailing stop
        /// moves, so the closing price is whatever the trail had ratcheted to,
        /// and rebuilding it from the original stop would understate every
        /// trailed win. The kernel that knows the level now reports it.
        ///
        /// `0.0` when there was no exit, which the `exit_bar < 0` case already
        /// signals.
        pub exit_price: f64,
        /// Realised P&L in account currency, carry and conversion applied.
        pub pnl: f64,
        /// `pnl` over the trade's initial risk.
        pub r_multiple: f64,
    }

    impl Default for NeoPopulationOutcome {
        fn default() -> Self {
            Self {
                candidate_id: 0,
                scenario_id: 0,
                exit_bar: -1,
                exit_reason: POPULATION_EXIT_NONE,
                entry_bar: -1,
                _pad: 0,
                mfe: 0.0,
                mae: 0.0,
                exit_price: 0.0,
                pnl: 0.0,
                r_multiple: 0.0,
            }
        }
    }

    /// Raw canonical level-10 metric row. Slot seven is the monthly hit rate.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
    pub struct NeoPopulationMetricRow {
        pub candidate_id: u64,
        pub scenario_id: u64,
        pub values: [f64; 11],
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct NeoPopulationCounters {
        pub event_count: u64,
        pub accepted_trade_count: u64,
        pub kernel_submissions: u64,
        pub synchronization_events: u64,
        pub dataset_upload_bytes: u64,
        pub gene_upload_bytes: u64,
        pub scenario_upload_bytes: u64,
        pub compact_readback_bytes: u64,
        pub full_readback_bytes: u64,
        pub reserved: [u64; 3],
    }

    const _: () = {
        use core::mem::{align_of, offset_of, size_of};

        assert!(size_of::<BufferRef>() == 16);
        assert!(align_of::<BufferRef>() == 8);
        assert!(size_of::<HandleToken>() == 32);
        assert!(offset_of!(HandleToken, generation) == 16);

        assert!(size_of::<DatasetHeader>() == 152);
        assert!(align_of::<DatasetHeader>() == 8);
        assert!(offset_of!(DatasetHeader, timestamps) == 24);
        assert!(offset_of!(DatasetHeader, features) == 104);
        assert!(offset_of!(DatasetHeader, days) == 136);

        assert!(size_of::<GeneDescriptor>() == 56);
        assert!(offset_of!(GeneDescriptor, stop_ticks) == 24);
        assert!(offset_of!(GeneDescriptor, reserved) == 48);

        assert!(size_of::<ScenarioDescriptor>() == 72);
        assert!(offset_of!(ScenarioDescriptor, commission_micros) == 48);
        assert!(offset_of!(ScenarioDescriptor, reserved) == 64);

        assert!(size_of::<TradeOutcome>() == 56);
        assert!(size_of::<Metrics>() == 80);
        assert!(size_of::<PropFirmState>() == 56);

        assert!(size_of::<NeoPopulationSettings>() == 184);
        assert!(offset_of!(NeoPopulationSettings, spread_pips_asian) == 160);
        assert!(offset_of!(NeoPopulationSettings, spread_pips_late_ny) == 176);
        assert!(offset_of!(NeoPopulationSettings, trailing_enabled) == 128);
        assert!(offset_of!(NeoPopulationSettings, trailing_min_lock_pips) == 152);
        assert!(align_of::<NeoPopulationSettings>() == 8);
        assert!(offset_of!(NeoPopulationSettings, gap_threshold_ms) == 24);
        assert!(offset_of!(NeoPopulationSettings, initial_equity) == 32);
        assert!(offset_of!(NeoPopulationSettings, adaptive_rr) == 120);

        assert!(size_of::<NeoPopulationEvent>() == 56);
        assert!(align_of::<NeoPopulationEvent>() == 8);
        assert!(offset_of!(NeoPopulationEvent, direction) == 24);
        assert!(offset_of!(NeoPopulationEvent, stop_price) == 32);

        assert!(size_of::<NeoPopulationOutcome>() == 72);
        assert!(align_of::<NeoPopulationOutcome>() == 8);
        assert!(offset_of!(NeoPopulationOutcome, exit_bar) == 16);

        assert!(size_of::<NeoPopulationMetricRow>() == 104);
        assert!(align_of::<NeoPopulationMetricRow>() == 8);
        assert!(offset_of!(NeoPopulationMetricRow, values) == 16);

        assert!(size_of::<NeoPopulationCounters>() == 96);
        assert!(align_of::<NeoPopulationCounters>() == 8);
        assert!(offset_of!(NeoPopulationCounters, reserved) == 72);
    };
}

pub mod scenario_rng {
    //! Counter-based Philox4x32-10 host reference used to specify device RNG.

    const M0: u32 = 0xD251_1F53;
    const M1: u32 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;

    pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
        let mut counter = counter;
        let mut key = key;
        for _ in 0..10 {
            counter = round(counter, key);
            key[0] = key[0].wrapping_add(W0);
            key[1] = key[1].wrapping_add(W1);
        }
        counter
    }

    pub fn counter_for(base_candidate_id: u64, scenario_id: u64, counter: u64) -> [u32; 4] {
        [
            base_candidate_id as u32,
            (base_candidate_id >> 32) as u32,
            scenario_id as u32 ^ counter as u32,
            (scenario_id >> 32) as u32 ^ (counter >> 32) as u32,
        ]
    }

    fn round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
        let p0 = (M0 as u64) * (counter[0] as u64);
        let p1 = (M1 as u64) * (counter[2] as u64);
        [
            (p1 >> 32) as u32 ^ counter[1] ^ key[0],
            p1 as u32,
            (p0 >> 32) as u32 ^ counter[3] ^ key[1],
            p0 as u32,
        ]
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn same_identity_produces_same_words() {
            let counter = counter_for(11, 22, 33);
            assert_eq!(
                philox4x32_10(counter, [44, 55]),
                philox4x32_10(counter, [44, 55])
            );
        }

        #[test]
        fn scenario_identity_changes_stream() {
            let first = philox4x32_10(counter_for(11, 22, 33), [44, 55]);
            let second = philox4x32_10(counter_for(11, 23, 33), [44, 55]);
            assert_ne!(first, second);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ABI_VERSION;
    use super::device::*;

    #[test]
    fn dataset_header_carries_shared_abi_version() {
        let header = DatasetHeader {
            abi_version: ABI_VERSION,
            ..DatasetHeader::default()
        };
        assert_eq!(header.abi_version, 3);
    }

    #[test]
    fn handle_token_identity_is_explicit() {
        let token = HandleToken {
            session_id: 7,
            backend_id: 2,
            device_id: 1,
            generation: 9,
            buffer_kind: 4,
            reserved: 0,
        };
        assert_ne!(token.session_id, token.generation);
    }
}
