use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use ndarray::Array2;
use polars::prelude::*;
use talib_sys::*;

#[derive(Debug, Clone)]
pub struct Ohlcv {
    pub timestamp: Option<Vec<i64>>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Option<Vec<f64>>,
}

impl Ohlcv {
    pub fn len(&self) -> usize {
        self.close.len()
    }

    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

fn is_sorted_timestamps(ts: &[i64]) -> bool {
    ts.windows(2).all(|w| w[0] <= w[1])
}

fn sort_ohlcv_by_timestamp(ohlcv: &Ohlcv) -> Ohlcv {
    let Some(ts) = &ohlcv.timestamp else {
        return ohlcv.clone();
    };
    if ts.len() != ohlcv.len() || is_sorted_timestamps(ts) {
        return ohlcv.clone();
    }
    let mut idx: Vec<usize> = (0..ts.len()).collect();
    idx.sort_by_key(|&i| ts[i]);

    let reorder = |src: &Vec<f64>| idx.iter().map(|&i| src[i]).collect::<Vec<f64>>();
    let sorted_ts = idx.iter().map(|&i| ts[i]).collect::<Vec<i64>>();
    let volume = ohlcv
        .volume
        .as_ref()
        .map(|v| idx.iter().map(|&i| v[i]).collect::<Vec<f64>>());

    Ohlcv {
        timestamp: Some(sorted_ts),
        open: reorder(&ohlcv.open),
        high: reorder(&ohlcv.high),
        low: reorder(&ohlcv.low),
        close: reorder(&ohlcv.close),
        volume,
    }
}

fn sort_dedup_ohlcv_by_timestamp(ohlcv: &Ohlcv) -> Ohlcv {
    let sorted = sort_ohlcv_by_timestamp(ohlcv);
    let Some(ts) = &sorted.timestamp else {
        return sorted;
    };
    if ts.len() != sorted.len() || ts.is_empty() {
        return sorted;
    }

    let mut keep: Vec<usize> = Vec::with_capacity(ts.len());
    let mut last_ts: Option<i64> = None;
    for (idx, &timestamp) in ts.iter().enumerate() {
        if last_ts != Some(timestamp) {
            keep.push(idx);
            last_ts = Some(timestamp);
        }
    }
    if keep.len() == ts.len() {
        return sorted;
    }

    let reorder = |src: &Vec<f64>| keep.iter().map(|&i| src[i]).collect::<Vec<f64>>();
    let dedup_ts = keep.iter().map(|&i| ts[i]).collect::<Vec<i64>>();
    let volume = sorted
        .volume
        .as_ref()
        .map(|v| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>());

    Ohlcv {
        timestamp: Some(dedup_ts),
        open: reorder(&sorted.open),
        high: reorder(&sorted.high),
        low: reorder(&sorted.low),
        close: reorder(&sorted.close),
        volume,
    }
}

fn tail_ohlcv(ohlcv: &Ohlcv, rows: usize) -> Ohlcv {
    if rows == 0 || ohlcv.len() <= rows {
        return ohlcv.clone();
    }
    let start = ohlcv.len() - rows;
    Ohlcv {
        timestamp: ohlcv.timestamp.as_ref().map(|ts| ts[start..].to_vec()),
        open: ohlcv.open[start..].to_vec(),
        high: ohlcv.high[start..].to_vec(),
        low: ohlcv.low[start..].to_vec(),
        close: ohlcv.close[start..].to_vec(),
        volume: ohlcv.volume.as_ref().map(|v| v[start..].to_vec()),
    }
}

fn pad_vec(mut values: Vec<f64>, len: usize) -> Vec<f64> {
    if values.len() < len {
        values.resize(len, f64::NAN);
    } else if values.len() > len {
        values.truncate(len);
    }
    values
}

fn compute_smc_feature_columns(ohlcv: &Ohlcv) -> Vec<(String, Vec<f64>)> {
    let n = ohlcv.len();
    let mut ob = vec![0.0_f64; n];
    let mut fvg = vec![0.0_f64; n];
    let mut liq = vec![0.0_f64; n];
    let mut trend = vec![0.0_f64; n];
    let mut premium = vec![0.0_f64; n];
    let mut inducement = vec![0.0_f64; n];
    let mut bos = vec![0.0_f64; n];
    let mut choch = vec![0.0_f64; n];
    let mut eqh = vec![0.0_f64; n];
    let mut eql = vec![0.0_f64; n];
    let mut displacement = vec![0.0_f64; n];

    if n == 0 {
        return vec![
            ("smc_ob".to_string(), ob),
            ("smc_fvg".to_string(), fvg),
            ("smc_liq".to_string(), liq),
            ("smc_trend".to_string(), trend),
            ("smc_premium".to_string(), premium),
            ("smc_inducement".to_string(), inducement),
            ("smc_bos".to_string(), bos),
            ("smc_choch".to_string(), choch),
            ("smc_eqh".to_string(), eqh),
            ("smc_eql".to_string(), eql),
            ("smc_displacement".to_string(), displacement),
        ];
    }

    const TREND_LOOKBACK: usize = 12;
    const STRUCTURE_LOOKBACK: usize = 20;
    const EQUAL_LOOKBACK: usize = 20;
    const DISPLACEMENT_LOOKBACK: usize = 20;
    const DISPLACEMENT_MULT: f64 = 1.8;

    for i in 0..n {
        let high_i = ohlcv.high[i];
        let low_i = ohlcv.low[i];
        let open_i = ohlcv.open[i];
        let close_i = ohlcv.close[i];

        if i >= TREND_LOOKBACK {
            let d = close_i - ohlcv.close[i - TREND_LOOKBACK];
            trend[i] = if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            };
        } else if i >= 1 {
            let d = close_i - ohlcv.close[i - 1];
            trend[i] = if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            };
        }

        let range_i = (high_i - low_i).abs();
        if range_i > 1e-12 {
            let rel = (close_i - low_i) / range_i;
            premium[i] = if rel <= 0.5 { 1.0 } else { -1.0 };
        }

        if i >= 1 {
            let prev_open = ohlcv.open[i - 1];
            let prev_close = ohlcv.close[i - 1];
            let prev_high = ohlcv.high[i - 1];
            let prev_low = ohlcv.low[i - 1];

            let bull_ob = close_i > open_i && prev_close < prev_open && close_i >= prev_high;
            let bear_ob = close_i < open_i && prev_close > prev_open && close_i <= prev_low;
            ob[i] = if bull_ob {
                1.0
            } else if bear_ob {
                -1.0
            } else {
                0.0
            };

            let body = (close_i - open_i).abs();
            let upper_wick = high_i - open_i.max(close_i);
            let lower_wick = open_i.min(close_i) - low_i;
            if body > 1e-12 && ((upper_wick / body) > 2.0 || (lower_wick / body) > 2.0) {
                inducement[i] = 1.0;
            }

            if i >= DISPLACEMENT_LOOKBACK {
                let mut avg_body = 0.0_f64;
                for j in (i - DISPLACEMENT_LOOKBACK)..i {
                    avg_body += (ohlcv.close[j] - ohlcv.open[j]).abs();
                }
                avg_body /= DISPLACEMENT_LOOKBACK as f64;
                if avg_body > 1e-12 && body >= avg_body * DISPLACEMENT_MULT {
                    displacement[i] = if close_i > open_i {
                        1.0
                    } else if close_i < open_i {
                        -1.0
                    } else {
                        0.0
                    };
                }
            }
        }

        if i >= 2 {
            if low_i > ohlcv.high[i - 2] {
                fvg[i] = 1.0;
            } else if high_i < ohlcv.low[i - 2] {
                fvg[i] = -1.0;
            }
        }

        if i >= 3 {
            let prev_low = ohlcv.low[(i - 3)..i]
                .iter()
                .fold(f64::INFINITY, |a, b| a.min(*b));
            let prev_high = ohlcv.high[(i - 3)..i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            if low_i < prev_low && close_i > prev_low {
                liq[i] = 1.0;
            } else if high_i > prev_high && close_i < prev_high {
                liq[i] = -1.0;
            }
        }

        if i >= 2 {
            let lookback = STRUCTURE_LOOKBACK.min(i);
            let start = i - lookback;
            let prev_struct_high = ohlcv.high[start..i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            let prev_struct_low = ohlcv.low[start..i]
                .iter()
                .fold(f64::INFINITY, |a, b| a.min(*b));
            if close_i > prev_struct_high {
                bos[i] = 1.0;
            } else if close_i < prev_struct_low {
                bos[i] = -1.0;
            }
            if i >= 1 {
                if bos[i] > 0.0 && trend[i - 1] < 0.0 {
                    choch[i] = 1.0;
                } else if bos[i] < 0.0 && trend[i - 1] > 0.0 {
                    choch[i] = -1.0;
                }
            }
        }

        if i >= 1 {
            let lookback = EQUAL_LOOKBACK.min(i);
            let start = i - lookback;
            let mut atr_proxy = 0.0_f64;
            for j in start..=i {
                atr_proxy += (ohlcv.high[j] - ohlcv.low[j]).abs();
            }
            atr_proxy /= (i - start + 1) as f64;
            let tol = (atr_proxy * 0.1).max(1e-6);

            for j in start..i {
                if (high_i - ohlcv.high[j]).abs() <= tol {
                    eqh[i] = 1.0;
                    break;
                }
            }
            for j in start..i {
                if (low_i - ohlcv.low[j]).abs() <= tol {
                    eql[i] = 1.0;
                    break;
                }
            }
        }
    }

    vec![
        ("smc_ob".to_string(), ob),
        ("smc_fvg".to_string(), fvg),
        ("smc_liq".to_string(), liq),
        ("smc_trend".to_string(), trend),
        ("smc_premium".to_string(), premium),
        ("smc_inducement".to_string(), inducement),
        ("smc_bos".to_string(), bos),
        ("smc_choch".to_string(), choch),
        ("smc_eqh".to_string(), eqh),
        ("smc_eql".to_string(), eql),
        ("smc_displacement".to_string(), displacement),
    ]
}

static TALIB_INIT: OnceLock<Result<(), anyhow::Error>> = OnceLock::new();

fn ensure_talib_init() -> Result<()> {
    let init = TALIB_INIT.get_or_init(|| {
        let rc = unsafe { TA_Initialize() };
        if rc == TA_RetCode::TA_SUCCESS {
            Ok(())
        } else {
            Err(anyhow::anyhow!("TA-Lib init failed: {:?}", rc))
        }
    });
    match init {
        Ok(_) => Ok(()),
        Err(err) => Err(anyhow::anyhow!(err.to_string())),
    }
}

const TA_IN_PRICE_VOLUME: i32 = 0x00000010;
const TA_IN_PRICE_OPENINTEREST: i32 = 0x00000020;
const TA_IN_PRICE_TIMESTAMP: i32 = 0x00000040;

fn normalize_name(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut underscore = false;
    for ch in input.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            underscore = false;
        } else if !underscore {
            out.push('_');
            underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

unsafe fn cstr_to_string(ptr: *const std::os::raw::c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    CStr::from_ptr(ptr).to_string_lossy().to_string()
}

unsafe fn string_table_to_vec(table: *const TA_StringTable) -> Vec<String> {
    if table.is_null() {
        return Vec::new();
    }
    let size = (*table).size as usize;
    if size == 0 || (*table).string.is_null() {
        return Vec::new();
    }
    let entries = std::slice::from_raw_parts((*table).string, size);
    let mut out = Vec::with_capacity(size);
    for &ptr in entries {
        if ptr.is_null() {
            continue;
        }
        out.push(cstr_to_string(ptr));
    }
    out
}

fn list_talib_functions() -> Result<Vec<String>> {
    let mut group_table: *mut TA_StringTable = ptr::null_mut();
    let ret = unsafe { TA_GroupTableAlloc(&mut group_table) };
    if ret != TA_RetCode::TA_SUCCESS {
        bail!("TA_GroupTableAlloc failed: {:?}", ret);
    }
    let groups = unsafe { string_table_to_vec(group_table) };
    unsafe {
        TA_GroupTableFree(group_table);
    }

    let mut names: HashSet<String> = HashSet::new();
    for group in groups {
        let c_group = CString::new(group.as_bytes())
            .map_err(|_| anyhow::anyhow!("TA group name contains null byte"))?;
        let mut func_table: *mut TA_StringTable = ptr::null_mut();
        let ret = unsafe { TA_FuncTableAlloc(c_group.as_ptr(), &mut func_table) };
        if ret != TA_RetCode::TA_SUCCESS {
            continue;
        }
        for name in unsafe { string_table_to_vec(func_table) } {
            if !name.is_empty() {
                names.insert(name);
            }
        }
        unsafe {
            TA_FuncTableFree(func_table);
        }
    }

    let mut list: Vec<String> = names.into_iter().collect();
    list.sort();
    Ok(list)
}

fn resolve_real_input(param_name: &str, idx: usize, ohlcv: &Ohlcv) -> *const f64 {
    let lower = param_name.to_ascii_lowercase();
    if lower.contains("open") {
        return ohlcv.open.as_ptr();
    }
    if lower.contains("high") {
        return ohlcv.high.as_ptr();
    }
    if lower.contains("low") {
        return ohlcv.low.as_ptr();
    }
    if lower.contains("close") {
        return ohlcv.close.as_ptr();
    }
    if lower.contains("volume") {
        if let Some(volume) = &ohlcv.volume {
            return volume.as_ptr();
        }
    }
    match idx % 5 {
        0 => ohlcv.close.as_ptr(),
        1 => ohlcv.open.as_ptr(),
        2 => ohlcv.high.as_ptr(),
        3 => ohlcv.low.as_ptr(),
        _ => ohlcv
            .volume
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or_else(|| ohlcv.close.as_ptr()),
    }
}

fn build_integer_input(param_name: &str, idx: usize, ohlcv: &Ohlcv) -> Vec<TA_Integer> {
    let lower = param_name.to_ascii_lowercase();
    let source: &Vec<f64> = if lower.contains("volume") {
        ohlcv.volume.as_ref().unwrap_or(&ohlcv.close)
    } else if lower.contains("open") {
        &ohlcv.open
    } else if lower.contains("high") {
        &ohlcv.high
    } else if lower.contains("low") {
        &ohlcv.low
    } else if lower.contains("close") {
        &ohlcv.close
    } else {
        match idx % 4 {
            0 => &ohlcv.close,
            1 => &ohlcv.open,
            2 => &ohlcv.high,
            _ => &ohlcv.low,
        }
    };

    source.iter().map(|v| *v as TA_Integer).collect()
}

enum OutputBuffer {
    Real(Vec<TA_Real>),
    Int(Vec<TA_Integer>),
}

struct ParamHolderGuard(*mut TA_ParamHolder);

impl Drop for ParamHolderGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                TA_ParamHolderFree(self.0);
            }
        }
    }
}

fn unique_name(base: &str, counts: &mut HashMap<String, usize>) -> String {
    if let Some(entry) = counts.get_mut(base) {
        *entry += 1;
        format!("{base}_{}", *entry)
    } else {
        counts.insert(base.to_string(), 1);
        base.to_string()
    }
}

const CORE_FUNCTION_PREFIXES: &[&str] = &[
    "adx", "adxr", "aroon", "aroonosc", "atr", "natr", "trange", "rsi", "stoch", "stochf",
    "stochrsi", "willr", "cci", "mfi", "obv", "ad", "adosc", "macd", "macdext", "macdfix", "ppo",
    "roc", "rocp", "rocr", "mom", "trix", "minus_di", "plus_di", "minus_dm", "plus_dm", "ema",
    "sma", "wma", "tema", "kama", "bbands",
];

const COMPACT_FUNCTION_PREFIXES: &[&str] = &[
    "adx", "atr", "natr", "rsi", "stoch", "stochrsi", "cci", "mfi", "obv", "macd", "ppo", "roc",
    "mom", "ema", "sma", "kama", "bbands",
];

fn profile_prefixes(profile: FeatureProfile) -> Option<&'static [&'static str]> {
    match profile {
        FeatureProfile::Full => None,
        FeatureProfile::Core => Some(CORE_FUNCTION_PREFIXES),
        FeatureProfile::Compact => Some(COMPACT_FUNCTION_PREFIXES),
    }
}

fn profile_allows_function(profile: FeatureProfile, normalized_fn: &str) -> bool {
    let Some(prefixes) = profile_prefixes(profile) else {
        return true;
    };
    prefixes
        .iter()
        .any(|p| normalized_fn == *p || normalized_fn.starts_with(&format!("{p}_")))
}

fn compute_talib_indicators(
    ohlcv: &Ohlcv,
    profile: FeatureProfile,
    max_outputs: usize,
) -> Result<Vec<(String, Vec<f64>)>> {
    if ohlcv.is_empty() {
        bail!("empty OHLCV data");
    }
    ensure_talib_init()?;

    let len = ohlcv.len();
    let func_names = list_talib_functions()?;
    let mut out: Vec<(String, Vec<f64>)> = Vec::new();
    let mut name_counts: HashMap<String, usize> = HashMap::new();
    let max_outputs = if max_outputs == 0 {
        usize::MAX
    } else {
        max_outputs
    };

    'func_loop: for func_name in func_names {
        if out.len() >= max_outputs {
            break;
        }
        let func_lower = normalize_name(&func_name);
        if !profile_allows_function(profile, &func_lower) {
            continue;
        }
        let c_name = CString::new(func_name.as_bytes())
            .map_err(|_| anyhow::anyhow!("TA function name contains null byte"))?;
        let mut handle_ptr: *const TA_FuncHandle = ptr::null();
        let ret = unsafe { TA_GetFuncHandle(c_name.as_ptr(), &mut handle_ptr) };
        if ret != TA_RetCode::TA_SUCCESS || handle_ptr.is_null() {
            continue;
        }

        let mut info_ptr: *const TA_FuncInfo = ptr::null();
        let ret = unsafe { TA_GetFuncInfo(handle_ptr, &mut info_ptr) };
        if ret != TA_RetCode::TA_SUCCESS || info_ptr.is_null() {
            continue;
        }
        let info = unsafe { &*info_ptr };

        let mut params: *mut TA_ParamHolder = ptr::null_mut();
        let ret = unsafe { TA_ParamHolderAlloc(handle_ptr, &mut params) };
        if ret != TA_RetCode::TA_SUCCESS || params.is_null() {
            continue;
        }
        let _guard = ParamHolderGuard(params);

        let volume_ptr = ohlcv
            .volume
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(ptr::null());

        let mut int_inputs: Vec<Vec<TA_Integer>> = Vec::new();
        let mut skip = false;

        for input_idx in 0..info.nbInput {
            let mut input_info_ptr: *const TA_InputParameterInfo = ptr::null();
            let ret =
                unsafe { TA_GetInputParameterInfo(handle_ptr, input_idx, &mut input_info_ptr) };
            if ret != TA_RetCode::TA_SUCCESS || input_info_ptr.is_null() {
                skip = true;
                break;
            }
            let input_info = unsafe { &*input_info_ptr };
            match input_info.type_ {
                x if x == TA_InputParameterType_TA_Input_Price => {
                    let flags = input_info.flags;
                    if (flags & TA_IN_PRICE_VOLUME) != 0 && volume_ptr.is_null() {
                        skip = true;
                        break;
                    }
                    if (flags & TA_IN_PRICE_OPENINTEREST) != 0 {
                        skip = true;
                        break;
                    }
                    if (flags & TA_IN_PRICE_TIMESTAMP) != 0 {
                        skip = true;
                        break;
                    }
                    let ret = unsafe {
                        TA_SetInputParamPricePtr(
                            params,
                            input_idx,
                            ohlcv.open.as_ptr(),
                            ohlcv.high.as_ptr(),
                            ohlcv.low.as_ptr(),
                            ohlcv.close.as_ptr(),
                            volume_ptr,
                            ptr::null(),
                        )
                    };
                    if ret != TA_RetCode::TA_SUCCESS {
                        skip = true;
                        break;
                    }
                }
                x if x == TA_InputParameterType_TA_Input_Real => {
                    let param_name = unsafe { cstr_to_string(input_info.paramName) };
                    if param_name.to_ascii_lowercase().contains("volume") && volume_ptr.is_null() {
                        skip = true;
                        break;
                    }
                    let ptr_in = resolve_real_input(&param_name, input_idx as usize, ohlcv);
                    let ret = unsafe { TA_SetInputParamRealPtr(params, input_idx, ptr_in) };
                    if ret != TA_RetCode::TA_SUCCESS {
                        skip = true;
                        break;
                    }
                }
                x if x == TA_InputParameterType_TA_Input_Integer => {
                    let param_name = unsafe { cstr_to_string(input_info.paramName) };
                    int_inputs.push(build_integer_input(&param_name, input_idx as usize, ohlcv));
                    let ptr_in = int_inputs.last().unwrap().as_ptr();
                    let ret = unsafe { TA_SetInputParamIntegerPtr(params, input_idx, ptr_in) };
                    if ret != TA_RetCode::TA_SUCCESS {
                        skip = true;
                        break;
                    }
                }
                _ => {
                    skip = true;
                    break;
                }
            }
        }

        if skip {
            continue;
        }

        for opt_idx in 0..info.nbOptInput {
            let mut opt_info_ptr: *const TA_OptInputParameterInfo = ptr::null();
            let ret =
                unsafe { TA_GetOptInputParameterInfo(handle_ptr, opt_idx, &mut opt_info_ptr) };
            if ret != TA_RetCode::TA_SUCCESS || opt_info_ptr.is_null() {
                continue;
            }
            let opt_info = unsafe { &*opt_info_ptr };
            match opt_info.type_ {
                x if x == TA_OptInputParameterType_TA_OptInput_RealRange
                    || x == TA_OptInputParameterType_TA_OptInput_RealList =>
                {
                    let _ =
                        unsafe { TA_SetOptInputParamReal(params, opt_idx, opt_info.defaultValue) };
                }
                x if x == TA_OptInputParameterType_TA_OptInput_IntegerRange
                    || x == TA_OptInputParameterType_TA_OptInput_IntegerList =>
                {
                    let _ = unsafe {
                        TA_SetOptInputParamInteger(
                            params,
                            opt_idx,
                            opt_info.defaultValue as TA_Integer,
                        )
                    };
                }
                _ => {}
            }
        }

        let mut outputs: Vec<(String, OutputBuffer)> = Vec::new();
        for out_idx in 0..info.nbOutput {
            if out.len() + outputs.len() >= max_outputs {
                break;
            }
            let mut out_info_ptr: *const TA_OutputParameterInfo = ptr::null();
            let ret = unsafe { TA_GetOutputParameterInfo(handle_ptr, out_idx, &mut out_info_ptr) };
            if ret != TA_RetCode::TA_SUCCESS || out_info_ptr.is_null() {
                continue;
            }
            let out_info = unsafe { &*out_info_ptr };
            let param_name = unsafe { cstr_to_string(out_info.paramName) };
            let param_norm = normalize_name(&param_name);
            let base = if info.nbOutput <= 1 {
                func_lower.clone()
            } else if param_norm.is_empty() {
                format!("{func_lower}_out{}", out_idx)
            } else {
                format!("{func_lower}_{param_norm}")
            };
            let name = unique_name(&normalize_name(&base), &mut name_counts);

            match out_info.type_ {
                x if x == TA_OutputParameterType_TA_Output_Integer => {
                    let mut buf = vec![0 as TA_Integer; len];
                    let ret =
                        unsafe { TA_SetOutputParamIntegerPtr(params, out_idx, buf.as_mut_ptr()) };
                    if ret == TA_RetCode::TA_SUCCESS {
                        outputs.push((name, OutputBuffer::Int(buf)));
                    }
                }
                _ => {
                    let mut buf = vec![0.0 as TA_Real; len];
                    let ret =
                        unsafe { TA_SetOutputParamRealPtr(params, out_idx, buf.as_mut_ptr()) };
                    if ret == TA_RetCode::TA_SUCCESS {
                        outputs.push((name, OutputBuffer::Real(buf)));
                    }
                }
            }
        }

        if outputs.is_empty() {
            continue;
        }

        let mut out_beg: TA_Integer = 0;
        let mut out_nb: TA_Integer = 0;
        let ret = unsafe {
            TA_CallFunc(
                params,
                0,
                (len as TA_Integer).saturating_sub(1),
                &mut out_beg,
                &mut out_nb,
            )
        };
        if ret != TA_RetCode::TA_SUCCESS || out_nb <= 0 || out_beg < 0 {
            continue;
        }

        let out_beg = out_beg as usize;
        let out_nb = out_nb as usize;

        for (name, buf) in outputs {
            let mut series = vec![f64::NAN; len];
            for i in 0..out_nb {
                let idx = out_beg + i;
                if idx >= len {
                    break;
                }
                let value = match &buf {
                    OutputBuffer::Real(vals) => vals[i] as f64,
                    OutputBuffer::Int(vals) => vals[i] as f64,
                };
                series[idx] = value;
            }
            out.push((name, series));
            if out.len() >= max_outputs {
                break 'func_loop;
            }
        }
    }

    if out.is_empty() {
        bail!("TA-Lib produced no indicators");
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    pub data: Array2<f32>,
    pub names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FeatureFrame {
    pub timestamps: Vec<i64>,
    pub names: Vec<String>,
    pub data: Array2<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureProfile {
    Full,
    Core,
    Compact,
}

impl FeatureProfile {
    pub fn from_str(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "core" | "stable" | "robust" => Self::Core,
            "compact" | "small" | "lite" => Self::Compact,
            _ => Self::Full,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Core => "core",
            Self::Compact => "compact",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FeatureBuildOptions {
    pub base_profile: FeatureProfile,
    pub htf_profile: FeatureProfile,
    pub max_base_features: usize,
    pub max_htf_features: usize,
}

impl Default for FeatureBuildOptions {
    fn default() -> Self {
        Self {
            base_profile: FeatureProfile::Full,
            htf_profile: FeatureProfile::Full,
            max_base_features: 0,
            max_htf_features: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SymbolDataset {
    pub symbol: String,
    pub frames: HashMap<String, Ohlcv>,
}

#[derive(Debug, Clone)]
pub struct FeatureCache {
    pub dir: PathBuf,
    pub ttl_minutes: u64,
    pub enabled: bool,
}

impl FeatureCache {
    pub fn new(dir: impl AsRef<Path>, ttl_minutes: u64, enabled: bool) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            ttl_minutes,
            enabled,
        }
    }

    fn is_fresh(&self, path: &Path) -> bool {
        if !self.enabled {
            return false;
        }
        if self.ttl_minutes == 0 {
            return true;
        }
        let Ok(metadata) = std::fs::metadata(path) else {
            return false;
        };
        let Ok(modified) = metadata.modified() else {
            return false;
        };
        let Ok(elapsed) = modified.elapsed() else {
            return false;
        };
        elapsed.as_secs() <= self.ttl_minutes * 60
    }

    pub fn load(&self, key: &str) -> Result<Option<FeatureFrame>> {
        if !self.enabled {
            return Ok(None);
        }
        let mut path = self.dir.clone();
        path.push(format!("{key}.parquet"));
        if !path.exists() {
            return Ok(None);
        }
        if !self.is_fresh(&path) {
            return Ok(None);
        }
        let read_result: Result<FeatureFrame> = (|| {
            let file = std::fs::File::open(&path)
                .with_context(|| format!("failed to open cached feature parquet: {}", path.display()))?;
            let df = ParquetReader::new(file)
                .finish()
                .with_context(|| format!("failed to read cached feature parquet: {}", path.display()))?;
            df_to_feature_frame(&df)
                .with_context(|| format!("failed to decode cached feature frame: {}", path.display()))
        })();

        match read_result {
            Ok(frame) => Ok(Some(frame)),
            Err(_) => {
                let _ = std::fs::remove_file(&path);
                Ok(None)
            }
        }
    }

    pub fn store(&self, key: &str, frame: &FeatureFrame) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir)?;
        let mut path = self.dir.clone();
        path.push(format!("{key}.parquet"));
        let mut tmp_path = self.dir.clone();
        tmp_path.push(format!("{key}.parquet.tmp"));
        let mut df = feature_frame_to_df(frame)?;
        let file = std::fs::File::create(&tmp_path)?;
        ParquetWriter::new(file).finish(&mut df)?;
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
        std::fs::rename(&tmp_path, &path)?;
        Ok(())
    }
}

impl SymbolDataset {
    pub fn timeframe(&self, tf: &str) -> Option<&Ohlcv> {
        self.frames.get(tf)
    }

    pub fn timeframes(&self) -> Vec<String> {
        let mut out: Vec<String> = self.frames.keys().cloned().collect();
        out.sort();
        out
    }

    pub fn tail_rows(&self, rows: usize) -> Self {
        if rows == 0 {
            return self.clone();
        }
        let frames = self
            .frames
            .iter()
            .map(|(tf, ohlcv)| (tf.clone(), tail_ohlcv(ohlcv, rows)))
            .collect();
        Self {
            symbol: self.symbol.clone(),
            frames,
        }
    }
}

pub const MANDATORY_TFS: [&str; 21] = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15", "M20", "M30", "H1", "H2", "H3", "H4",
    "H6", "H8", "H12", "D1", "W1", "MN1",
];

fn series_to_f64(series: &Series) -> Result<Vec<f64>> {
    let casted = series.cast(&DataType::Float64)?;
    let chunked = casted.f64().context("series cast to f64 failed")?;
    Ok(chunked.into_iter().map(|v| v.unwrap_or(0.0)).collect())
}

fn series_to_f32(series: &Series, n_rows: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n_rows];
    let casted = series
        .cast(&DataType::Float64)
        .unwrap_or_else(|_| series.clone());
    if let Ok(chunked) = casted.f64() {
        for (i, v) in chunked.into_iter().enumerate().take(n_rows) {
            out[i] = v.unwrap_or(0.0) as f32;
        }
    }
    out
}

fn find_series(df: &DataFrame, candidates: &[&str]) -> Option<Series> {
    for name in df.get_column_names() {
        let lower = name.to_ascii_lowercase();
        if candidates.iter().any(|c| lower == *c) {
            // In polars 0.47, column() returns &Column, convert to Series
            return df
                .column(name)
                .ok()
                .map(|col| col.as_materialized_series().clone());
        }
    }
    None
}

fn extract_timestamps(df: &DataFrame) -> Result<Option<Vec<i64>>> {
    let series = match find_series(df, &["timestamp", "time", "datetime", "date"]) {
        Some(s) => s,
        None => return Ok(None),
    };
    let casted = series.cast(&DataType::Int64)?;
    let chunked = casted.i64().context("timestamp cast to i64 failed")?;
    Ok(Some(chunked.into_iter().map(|v| v.unwrap_or(0)).collect()))
}

fn feature_frame_to_df(frame: &FeatureFrame) -> Result<DataFrame> {
    let mut cols: Vec<Column> = Vec::with_capacity(frame.names.len() + 1);
    cols.push(Series::new("timestamp".into(), frame.timestamps.clone()).into());
    for (idx, name) in frame.names.iter().enumerate() {
        let mut col = Vec::with_capacity(frame.data.nrows());
        for row in 0..frame.data.nrows() {
            col.push(frame.data[(row, idx)]);
        }
        cols.push(Series::new(name.as_str().into(), col).into());
    }
    Ok(DataFrame::new(cols)?)
}

fn df_to_feature_frame(df: &DataFrame) -> Result<FeatureFrame> {
    let timestamps = extract_timestamps(df)?.context("cached features missing timestamp column")?;
    let n_rows = timestamps.len();
    let n_cols = df
        .get_columns()
        .iter()
        .filter(|col| {
            !col.as_materialized_series()
                .name()
                .eq_ignore_ascii_case("timestamp")
        })
        .count();
    let mut names = Vec::with_capacity(n_cols);
    let mut data = Array2::<f32>::zeros((n_rows, n_cols));
    let mut col_idx = 0usize;
    for col in df.get_columns() {
        let series = col.as_materialized_series();
        if series.name().eq_ignore_ascii_case("timestamp") {
            continue;
        }
        names.push(series.name().to_string());
        let vals = series_to_f32(series, n_rows);
        let len = vals.len().min(n_rows);
        for i in 0..len {
            data[(i, col_idx)] = vals[i];
        }
        col_idx += 1;
    }
    Ok(FeatureFrame {
        timestamps,
        names,
        data,
    })
}

pub fn load_parquet(path: impl AsRef<Path>) -> Result<Ohlcv> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open parquet file: {}", path.display()))?;
    let df = ParquetReader::new(file).finish()?;

    let timestamp = extract_timestamps(&df)?;
    let open = find_series(&df, &["open", "o"]).context("missing open column")?;
    let high = find_series(&df, &["high", "h"]).context("missing high column")?;
    let low = find_series(&df, &["low", "l"]).context("missing low column")?;
    let close = find_series(&df, &["close", "c"]).context("missing close column")?;
    let volume = find_series(&df, &["volume", "vol", "v"]);

    let open = series_to_f64(&open)?;
    let high = series_to_f64(&high)?;
    let low = series_to_f64(&low)?;
    let close = series_to_f64(&close)?;
    let volume = match volume {
        Some(ref series) => Some(series_to_f64(series)?),
        None => None,
    };

    let n = close.len();
    if open.len() != n || high.len() != n || low.len() != n {
        bail!("OHLC columns have mismatched lengths");
    }
    if let Some(ref vol) = volume {
        if vol.len() != n {
            bail!("volume column length does not match OHLC length");
        }
    }
    if let Some(ref ts) = timestamp {
        if ts.len() != n {
            bail!("timestamp column length does not match OHLC length");
        }
    }

    Ok(Ohlcv {
        timestamp,
        open,
        high,
        low,
        close,
        volume,
    })
}

pub fn load_symbol_dataset(root: impl AsRef<Path>, symbol: &str) -> Result<SymbolDataset> {
    let tfs = discover_timeframes(&root, symbol)?;
    if tfs.is_empty() {
        bail!("no timeframes discovered for symbol={}", symbol);
    }
    let mut frames = HashMap::new();
    for tf in tfs {
        let ohlcv = sort_dedup_ohlcv_by_timestamp(&load_symbol_timeframe(&root, symbol, &tf)?);
        frames.insert(tf, ohlcv);
    }
    Ok(SymbolDataset {
        symbol: symbol.to_string(),
        frames,
    })
}

pub fn load_symbol_dataset_with_timeframes(
    root: impl AsRef<Path>,
    symbol: &str,
    timeframes: &[&str],
) -> Result<SymbolDataset> {
    let mut frames = HashMap::new();
    for tf in timeframes {
        let ohlcv = sort_dedup_ohlcv_by_timestamp(&load_symbol_timeframe(&root, symbol, tf)?);
        frames.insert(tf.to_string(), ohlcv);
    }
    Ok(SymbolDataset {
        symbol: symbol.to_string(),
        frames,
    })
}

pub fn compute_talib_features(ohlcv: &Ohlcv) -> Result<FeatureMatrix> {
    let sorted = sort_dedup_ohlcv_by_timestamp(ohlcv);
    let n_rows = sorted.len();
    if n_rows == 0 {
        bail!("empty OHLCV data");
    }

    let indicators = compute_talib_indicators(&sorted, FeatureProfile::Full, 0)?;
    let smc_cols = compute_smc_feature_columns(&sorted);
    let n_cols = indicators.len() + smc_cols.len();
    let mut names = Vec::with_capacity(n_cols);
    let mut out = Array2::<f32>::zeros((n_rows, n_cols));
    for (col_idx, (name, values)) in indicators.into_iter().enumerate() {
        names.push(format!("ta_{name}"));
        let vals = pad_vec(values, n_rows);
        let len = vals.len().min(n_rows);
        for i in 0..len {
            out[(i, col_idx)] = vals[i] as f32;
        }
    }

    let mut col_idx = names.len();
    for (name, values) in smc_cols {
        names.push(name);
        let vals = pad_vec(values, n_rows);
        let len = vals.len().min(n_rows);
        for i in 0..len {
            out[(i, col_idx)] = vals[i] as f32;
        }
        col_idx += 1;
    }

    Ok(FeatureMatrix { data: out, names })
}

pub fn compute_talib_feature_frame_with_options(
    ohlcv: &Ohlcv,
    include_raw: bool,
    profile: FeatureProfile,
    max_features: usize,
) -> Result<FeatureFrame> {
    let sorted = sort_dedup_ohlcv_by_timestamp(ohlcv);
    let n_rows = sorted.len();
    if n_rows == 0 {
        bail!("empty OHLCV data");
    }

    let timestamps = sorted
        .timestamp
        .clone()
        .unwrap_or_else(|| (0..n_rows as i64).collect());

    let indicators = compute_talib_indicators(&sorted, profile, max_features)?;
    let smc_cols = compute_smc_feature_columns(&sorted);
    let raw_cols = if include_raw {
        4 + usize::from(sorted.volume.is_some())
    } else {
        0
    };
    let n_cols = raw_cols + indicators.len() + smc_cols.len();
    let mut names = Vec::with_capacity(n_cols);
    let mut out = Array2::<f32>::zeros((n_rows, n_cols));
    let mut col_idx = 0usize;

    if include_raw {
        names.push("open".to_string());
        for i in 0..n_rows {
            out[(i, col_idx)] = sorted.open[i] as f32;
        }
        col_idx += 1;

        names.push("high".to_string());
        for i in 0..n_rows {
            out[(i, col_idx)] = sorted.high[i] as f32;
        }
        col_idx += 1;

        names.push("low".to_string());
        for i in 0..n_rows {
            out[(i, col_idx)] = sorted.low[i] as f32;
        }
        col_idx += 1;

        names.push("close".to_string());
        for i in 0..n_rows {
            out[(i, col_idx)] = sorted.close[i] as f32;
        }
        col_idx += 1;

        if let Some(volume) = &sorted.volume {
            names.push("volume".to_string());
            for i in 0..n_rows {
                out[(i, col_idx)] = volume[i] as f32;
            }
            col_idx += 1;
        }
    }

    for (name, values) in indicators {
        names.push(format!("ta_{name}"));
        let vals = pad_vec(values, n_rows);
        let len = vals.len().min(n_rows);
        for i in 0..len {
            out[(i, col_idx)] = vals[i] as f32;
        }
        col_idx += 1;
    }

    for (name, values) in smc_cols {
        names.push(name);
        let vals = pad_vec(values, n_rows);
        let len = vals.len().min(n_rows);
        for i in 0..len {
            out[(i, col_idx)] = vals[i] as f32;
        }
        col_idx += 1;
    }

    Ok(FeatureFrame {
        timestamps,
        names,
        data: out,
    })
}

pub fn compute_talib_feature_frame(ohlcv: &Ohlcv, include_raw: bool) -> Result<FeatureFrame> {
    compute_talib_feature_frame_with_options(ohlcv, include_raw, FeatureProfile::Full, 0)
}

fn select_htf_indices(
    names: &[String],
    profile: FeatureProfile,
    max_features: usize,
) -> Vec<usize> {
    let mut out = Vec::new();
    let mut count = 0usize;
    let cap = if max_features == 0 {
        usize::MAX
    } else {
        max_features
    };
    for (idx, name) in names.iter().enumerate() {
        let mut keep = true;
        if name.to_ascii_lowercase().starts_with("smc_") {
            keep = true;
        } else if !name.eq_ignore_ascii_case("open")
            && !name.eq_ignore_ascii_case("high")
            && !name.eq_ignore_ascii_case("low")
            && !name.eq_ignore_ascii_case("close")
            && !name.eq_ignore_ascii_case("volume")
        {
            let norm = normalize_name(name.strip_prefix("ta_").unwrap_or(name));
            keep = profile_allows_function(profile, &norm);
        }
        if !keep {
            continue;
        }
        out.push(idx);
        count += 1;
        if count >= cap {
            break;
        }
    }
    out
}

fn select_columns(data: &Array2<f32>, indices: &[usize]) -> Array2<f32> {
    let n_rows = data.nrows();
    let mut out = Array2::<f32>::zeros((n_rows, indices.len()));
    for (col_pos, col_idx) in indices.iter().enumerate() {
        for row in 0..n_rows {
            out[(row, col_pos)] = data[(row, *col_idx)];
        }
    }
    out
}

fn align_features(base_ts: &[i64], htf_ts: &[i64], htf_data: &Array2<f32>) -> Array2<f32> {
    let n_base = base_ts.len();
    let n_htf = htf_ts.len();
    let n_cols = htf_data.ncols();
    let mut out = Array2::<f32>::zeros((n_base, n_cols));
    if n_htf == 0 || n_base == 0 {
        return out;
    }
    let mut j = 0usize;
    for i in 0..n_base {
        let target = base_ts[i];
        while j + 1 < n_htf && htf_ts[j + 1] <= target {
            j += 1;
        }
        if htf_ts[j] > target {
            continue;
        }
        if j == 0 {
            continue;
        }
        let src = j - 1;
        for c in 0..n_cols {
            out[(i, c)] = htf_data[(src, c)];
        }
    }
    out
}

fn hstack(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let (rows, cols_a) = a.dim();
    let cols_b = b.ncols();
    let mut out = Array2::<f32>::zeros((rows, cols_a + cols_b));
    for r in 0..rows {
        for c in 0..cols_a {
            out[(r, c)] = a[(r, c)];
        }
        for c in 0..cols_b {
            out[(r, cols_a + c)] = b[(r, c)];
        }
    }
    out
}

pub fn missing_timeframes(dataset: &SymbolDataset, required: &[&str]) -> Vec<String> {
    let mut missing = Vec::new();
    for tf in required {
        if !dataset.frames.contains_key(*tf) {
            missing.push((*tf).to_string());
        }
    }
    missing
}

pub fn ensure_timeframes(dataset: &SymbolDataset, required: &[&str]) -> Result<()> {
    let missing = missing_timeframes(dataset, required);
    if !missing.is_empty() {
        bail!("missing timeframes: {}", missing.join(", "));
    }
    Ok(())
}

fn timeframe_to_ms(tf: &str) -> Option<i64> {
    match tf.to_ascii_uppercase().as_str() {
        "M1" => Some(60_000),
        "M2" => Some(120_000),
        "M3" => Some(180_000),
        "M4" => Some(240_000),
        "M5" => Some(300_000),
        "M6" => Some(360_000),
        "M10" => Some(600_000),
        "M12" => Some(720_000),
        "M15" => Some(900_000),
        "M20" => Some(1_200_000),
        "M30" => Some(1_800_000),
        "H1" => Some(3_600_000),
        "H2" => Some(7_200_000),
        "H3" => Some(10_800_000),
        "H4" => Some(14_400_000),
        "H6" => Some(21_600_000),
        "H8" => Some(28_800_000),
        "H12" => Some(43_200_000),
        "D1" => Some(86_400_000),
        "W1" => Some(604_800_000),
        "MN1" => Some(2_592_000_000),
        _ => None,
    }
}

pub fn resample_ohlcv(ohlcv: &Ohlcv, target_tf: &str) -> Result<Ohlcv> {
    let Some(ts) = ohlcv.timestamp.clone() else {
        bail!("cannot resample without timestamps");
    };
    let Some(bucket_ms) = timeframe_to_ms(target_tf) else {
        bail!("unsupported timeframe: {}", target_tf);
    };
    if ts.is_empty() {
        bail!("empty timestamp series");
    }

    let mut out_ts = Vec::new();
    let mut out_open = Vec::new();
    let mut out_high = Vec::new();
    let mut out_low = Vec::new();
    let mut out_close = Vec::new();
    let mut out_vol: Option<Vec<f64>> = ohlcv.volume.as_ref().map(|_| Vec::new());

    let mut current_bucket = ts[0] / bucket_ms;
    let mut open = ohlcv.open[0];
    let mut high = ohlcv.high[0];
    let mut low = ohlcv.low[0];
    let mut close = ohlcv.close[0];
    let mut volume = ohlcv.volume.as_ref().map(|v| v[0]).unwrap_or(0.0);

    for i in 1..ts.len() {
        let bucket = ts[i] / bucket_ms;
        if bucket != current_bucket {
            out_ts.push(ts[i - 1]);
            out_open.push(open);
            out_high.push(high);
            out_low.push(low);
            out_close.push(close);
            if let Some(ref mut vec) = out_vol {
                vec.push(volume);
            }

            current_bucket = bucket;
            open = ohlcv.open[i];
            high = ohlcv.high[i];
            low = ohlcv.low[i];
            close = ohlcv.close[i];
            volume = ohlcv.volume.as_ref().map(|v| v[i]).unwrap_or(0.0);
        } else {
            if ohlcv.high[i] > high {
                high = ohlcv.high[i];
            }
            if ohlcv.low[i] < low {
                low = ohlcv.low[i];
            }
            close = ohlcv.close[i];
            volume += ohlcv.volume.as_ref().map(|v| v[i]).unwrap_or(0.0);
        }
    }

    out_ts.push(*ts.last().unwrap());
    out_open.push(open);
    out_high.push(high);
    out_low.push(low);
    out_close.push(close);
    if let Some(ref mut vec) = out_vol {
        vec.push(volume);
    }

    Ok(Ohlcv {
        timestamp: Some(out_ts),
        open: out_open,
        high: out_high,
        low: out_low,
        close: out_close,
        volume: out_vol,
    })
}

pub fn ensure_timeframes_with_resample(
    dataset: &SymbolDataset,
    base_tf: &str,
    targets: &[&str],
) -> Result<SymbolDataset> {
    let base = dataset
        .frames
        .get(base_tf)
        .context("base timeframe missing for resample")?;
    let mut frames = dataset.frames.clone();
    for tf in targets {
        if frames.contains_key(*tf) {
            continue;
        }
        let resampled = resample_ohlcv(base, tf)?;
        frames.insert((*tf).to_string(), resampled);
    }
    Ok(SymbolDataset {
        symbol: dataset.symbol.clone(),
        frames,
    })
}

pub fn prepare_multitimeframe_features(
    dataset: &SymbolDataset,
    base_tf: &str,
    higher_tfs: &[&str],
    cache: Option<&FeatureCache>,
) -> Result<FeatureFrame> {
    prepare_multitimeframe_features_with_options(
        dataset,
        base_tf,
        higher_tfs,
        cache,
        &FeatureBuildOptions::default(),
    )
}

pub fn prepare_multitimeframe_features_with_options(
    dataset: &SymbolDataset,
    base_tf: &str,
    higher_tfs: &[&str],
    cache: Option<&FeatureCache>,
    options: &FeatureBuildOptions,
) -> Result<FeatureFrame> {
    let base_tf = if dataset.frames.contains_key(base_tf) {
        base_tf.to_string()
    } else if dataset.frames.contains_key("M5") {
        "M5".to_string()
    } else if dataset.frames.contains_key("M1") {
        "M1".to_string()
    } else {
        dataset
            .frames
            .keys()
            .next()
            .cloned()
            .context("no timeframes available")?
    };

    let base_ohlcv = dataset
        .frames
        .get(&base_tf)
        .context("base timeframe data missing")?;

    let base_key = format!(
        "{}_{}_base_{}_{}",
        dataset.symbol,
        base_tf,
        options.base_profile.as_str(),
        options.max_base_features
    );
    let base_frame = if let Some(cache) = cache {
        if let Some(frame) = cache.load(&base_key)? {
            frame
        } else {
            let frame = compute_talib_feature_frame_with_options(
                base_ohlcv,
                true,
                options.base_profile,
                options.max_base_features,
            )?;
            cache.store(&base_key, &frame)?;
            frame
        }
    } else {
        compute_talib_feature_frame_with_options(
            base_ohlcv,
            true,
            options.base_profile,
            options.max_base_features,
        )?
    };

    let base_ts = base_frame.timestamps.clone();
    let mut names = base_frame.names.clone();
    let mut data = base_frame.data.clone();

    let mut targets: Vec<String> = if higher_tfs.is_empty() {
        dataset
            .frames
            .keys()
            .filter(|tf| *tf != &base_tf)
            .cloned()
            .collect()
    } else {
        higher_tfs.iter().map(|tf| tf.to_string()).collect()
    };
    targets.sort();

    for tf in targets {
        if tf == base_tf {
            continue;
        }
        let htf_ohlcv = match dataset.frames.get(&tf) {
            Some(val) => val,
            None => continue,
        };
        let htf_key = format!(
            "{}_{}_htf_{}_{}",
            dataset.symbol,
            tf,
            options.htf_profile.as_str(),
            options.max_htf_features
        );
        let htf_frame = if let Some(cache) = cache {
            if let Some(frame) = cache.load(&htf_key)? {
                frame
            } else {
                let frame = compute_talib_feature_frame_with_options(
                    htf_ohlcv,
                    false,
                    options.htf_profile,
                    options.max_htf_features,
                )?;
                cache.store(&htf_key, &frame)?;
                frame
            }
        } else {
            compute_talib_feature_frame_with_options(
                htf_ohlcv,
                false,
                options.htf_profile,
                options.max_htf_features,
            )?
        };

        if htf_frame.timestamps.is_empty() {
            continue;
        }
        let indices = select_htf_indices(
            &htf_frame.names,
            options.htf_profile,
            options.max_htf_features,
        );
        if indices.is_empty() {
            continue;
        }
        let subset = select_columns(&htf_frame.data, &indices);
        let aligned = align_features(&base_ts, &htf_frame.timestamps, &subset);
        let prefixed_names: Vec<String> = indices
            .iter()
            .map(|idx| format!("{}_{}", tf, htf_frame.names[*idx]))
            .collect();

        data = hstack(&data, &aligned);
        names.extend(prefixed_names);
    }

    Ok(FeatureFrame {
        timestamps: base_ts,
        names,
        data,
    })
}

pub fn load_symbol_timeframe(
    root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
) -> Result<Ohlcv> {
    let mut path = PathBuf::from(root.as_ref());
    path.push(format!("symbol={}", symbol));
    path.push(format!("timeframe={}", timeframe));
    path.push("data.parquet");
    load_parquet(&path)
}

pub fn discover_symbols(root: impl AsRef<Path>) -> Result<Vec<String>> {
    let root = root.as_ref();
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(symbol) = name.strip_prefix("symbol=") {
            out.push(symbol.to_string());
        }
    }
    out.sort();
    Ok(out)
}

pub fn discover_timeframes(root: impl AsRef<Path>, symbol: &str) -> Result<Vec<String>> {
    let mut path = PathBuf::from(root.as_ref());
    path.push(format!("symbol={}", symbol));
    let mut out = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(tf) = name.strip_prefix("timeframe=") {
            out.push(tf.to_string());
        }
    }
    out.sort();
    Ok(out)
}
