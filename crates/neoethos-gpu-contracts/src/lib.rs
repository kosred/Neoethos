//! Backend-independent contracts for the NeoEthos GPU-native discovery pipeline.
//!
//! This crate has no GPU runtime dependency. It separates ergonomic host DTOs
//! from stable C-compatible POD layouts shared by CubeCL, native CUDA and Rust
//! FFI callers.

use serde::{Deserialize, Serialize};

pub const ABI_VERSION: u32 = 1;

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
    use super::device::*;
    use super::ABI_VERSION;

    #[test]
    fn dataset_header_carries_shared_abi_version() {
        let header = DatasetHeader {
            abi_version: ABI_VERSION,
            ..DatasetHeader::default()
        };
        assert_eq!(header.abi_version, 1);
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
