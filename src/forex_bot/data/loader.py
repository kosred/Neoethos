from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import Settings

logger = logging.getLogger(__name__)
_RUST_DATA_BACKEND_OK: bool | None = None
_RUST_DATA_WARNED_UNAVAILABLE = False
_OHLCV_COLUMN_ALIASES = {
    "timestamp",
    "time",
    "datetime",
    "date",
    "o",
    "open",
    "h",
    "high",
    "l",
    "low",
    "c",
    "close",
    "v",
    "vol",
    "volume",
}


def _rust_data_backend_available(*, force_log: bool = False) -> bool:
    global _RUST_DATA_BACKEND_OK, _RUST_DATA_WARNED_UNAVAILABLE
    if _RUST_DATA_BACKEND_OK is None:
        try:
            import forex_bindings  # type: ignore

            _RUST_DATA_BACKEND_OK = hasattr(forex_bindings, "load_symbol_frames")
        except Exception:
            _RUST_DATA_BACKEND_OK = False
    if force_log and not _RUST_DATA_BACKEND_OK and not _RUST_DATA_WARNED_UNAVAILABLE:
        logger.warning(
            "Rust data backend requested but forex_bindings.load_symbol_frames is unavailable; using Python backend."
        )
        _RUST_DATA_WARNED_UNAVAILABLE = True
    return bool(_RUST_DATA_BACKEND_OK)


def _disable_rust_data_backend() -> None:
    global _RUST_DATA_BACKEND_OK
    _RUST_DATA_BACKEND_OK = False


def _use_rust_data_backend() -> bool:
    raw = os.environ.get("FOREX_BOT_RUST_DATA")
    if raw is not None and str(raw).strip() != "":
        mode = str(raw).strip().lower()
        if mode in {"auto", "detect"}:
            return _rust_data_backend_available()
        enabled = mode in {"1", "true", "yes", "on", "rust"}
        return enabled and _rust_data_backend_available(force_log=True)
    mode = str(os.environ.get("FOREX_BOT_DATA_BACKEND", "auto")).strip().lower()
    if mode in {"rust", "rs", "1", "true", "yes", "on"}:
        return _rust_data_backend_available(force_log=True)
    if mode in {"python", "py", "0", "false", "no", "off"}:
        return False
    return _rust_data_backend_available()


def _timeframe_to_freq(tf: str) -> str | None:
    tf = str(tf or "").upper()
    return {
        "M1": "1min",
        "M2": "2min",
        "M3": "3min",
        "M4": "4min",
        "M5": "5min",
        "M6": "6min",
        "M10": "10min",
        "M12": "12min",
        "M15": "15min",
        "M20": "20min",
        "M30": "30min",
        "H1": "1H",
        "H2": "2H",
        "H3": "3H",
        "H4": "4H",
        "H6": "6H",
        "H8": "8H",
        "H12": "12H",
        "D1": "1D",
        "W1": "1W",
        "MN1": "1M",
    }.get(tf)


def _timeframe_to_minutes(tf: str) -> int | None:
    tf = str(tf or "").upper()
    return {
        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M4": 4,
        "M5": 5,
        "M6": 6,
        "M10": 10,
        "M12": 12,
        "M15": 15,
        "M20": 20,
        "M30": 30,
        "H1": 60,
        "H2": 120,
        "H3": 180,
        "H4": 240,
        "H6": 360,
        "H8": 480,
        "H12": 720,
        "D1": 1440,
        "W1": 10080,
        "MN1": 43200,
    }.get(tf)


def _ordered_timeframes(tfs: list[str]) -> list[str]:
    uniq = [str(tf or "").upper() for tf in tfs if str(tf or "").strip()]
    seen: set[str] = set()
    out: list[str] = []
    for tf in uniq:
        if tf in seen:
            continue
        seen.add(tf)
        out.append(tf)

    def _key(tf: str) -> tuple[int, str]:
        mins = _timeframe_to_minutes(tf)
        if mins is None:
            return (10**9, tf)
        return (int(mins), tf)

    return sorted(out, key=_key)


def _resample_missing_from_available(frames: dict[str, pd.DataFrame], required_tfs: list[str]) -> dict[str, pd.DataFrame]:
    if not frames:
        return frames
    required = _ordered_timeframes(required_tfs)
    if not required:
        return frames

    available = {str(k).upper(): v for k, v in frames.items() if isinstance(v, pd.DataFrame) and not v.empty}
    if not available:
        return frames

    # Prefer the finest available frame as source for upsampling to larger bars.
    source_order = _ordered_timeframes(list(available.keys()))
    if not source_order:
        return frames
    source_tf = source_order[0]
    source_df = available.get(source_tf)
    if source_df is None or source_df.empty:
        return frames

    for tf in required:
        if tf in available:
            continue
        resampled = _resample_ohlcv(source_df, tf)
        if resampled is None or resampled.empty:
            continue
        try:
            resampled.attrs["source"] = str(source_df.attrs.get("source", "resampled"))
        except Exception:
            pass
        frames[tf] = resampled
        available[tf] = resampled

    return frames


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    rename = {}
    for col in out.columns:
        low = str(col).lower()
        if low in {"o", "open"}:
            rename[col] = "open"
        elif low in {"h", "high"}:
            rename[col] = "high"
        elif low in {"l", "low"}:
            rename[col] = "low"
        elif low in {"c", "close"}:
            rename[col] = "close"
        elif low in {"v", "vol", "volume"}:
            rename[col] = "volume"
        elif low in {"timestamp", "time", "datetime", "date"}:
            rename[col] = "timestamp"
    if rename:
        out = out.rename(columns=rename)
    return out


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "close" not in out.columns:
        return out
    for col in ("open", "high", "low"):
        if col not in out.columns:
            out[col] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = 0.0
    return out


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "timestamp" in out.columns:
        idx = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.set_index(idx)
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def _read_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        import polars as pl

        if path.suffix.lower() == ".parquet":
            cols: list[str] | None = None
            try:
                schema = pl.scan_parquet(path).collect_schema()
                keep = [name for name in schema.names() if str(name).lower() in _OHLCV_COLUMN_ALIASES]
                cols = keep or None
            except Exception:
                cols = None
            df = pl.read_parquet(path, columns=cols).to_pandas()
        else:
            df = pl.read_csv(path).to_pandas()
        return df
    except Exception:
        try:
            if path.suffix.lower() == ".parquet":
                cols: list[str] | None = None
                try:
                    import pyarrow.parquet as pq

                    names = [str(n) for n in pq.ParquetFile(path).schema.names]
                    keep = [name for name in names if str(name).lower() in _OHLCV_COLUMN_ALIASES]
                    cols = keep or None
                except Exception:
                    cols = None
                return pd.read_parquet(path, columns=cols)
            return pd.read_csv(path)
        except Exception as exc:
            logger.warning("Failed to read data file %s: %s", path, exc)
            return None


def _resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    freq = _timeframe_to_freq(tf)
    if freq is None:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(freq).agg(agg).dropna()
    return out


class _StubMT5Connection:
    async def get_account_information(self) -> dict[str, Any]:
        return {}

    async def positions_get(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return []

    async def get_history_deals(self, _from: int, _to: int) -> list[dict[str, Any]]:
        return []

    async def get_symbol_info(self, _symbol: str) -> dict[str, Any]:
        return {}

    async def get_symbol_price(self, _symbol: str) -> dict[str, Any]:
        return {}

    async def create_market_buy_order(self, *args, **kwargs) -> dict[str, Any]:
        return {"success": False, "reason": "MT5 not connected"}

    async def create_market_sell_order(self, *args, **kwargs) -> dict[str, Any]:
        return {"success": False, "reason": "MT5 not connected"}

    async def close_position_by_ticket(self, *args, **kwargs) -> dict[str, Any]:
        return {"success": False, "reason": "MT5 not connected"}


def _mt5_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "_asdict"):
        try:
            return obj._asdict()
        except Exception:
            return {}
    try:
        return dict(obj)
    except Exception:
        return {}


def _mt5_list_to_dict(items: Any) -> list[dict[str, Any]]:
    if items is None:
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        out.append(_mt5_to_dict(item))
    return out


class _MT5Connection:
    def __init__(self, mt5_module: Any) -> None:
        self.mt5 = mt5_module

    async def get_account_information(self) -> dict[str, Any]:
        return _mt5_to_dict(self.mt5.account_info())

    async def positions_get(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            return _mt5_list_to_dict(self.mt5.positions_get(symbol=symbol))
        return _mt5_list_to_dict(self.mt5.positions_get())

    async def get_history_deals(self, _from: int, _to: int) -> list[dict[str, Any]]:
        return _mt5_list_to_dict(self.mt5.history_deals_get(_from, _to))

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        return _mt5_to_dict(self.mt5.symbol_info(symbol))

    async def get_symbol_price(self, symbol: str) -> dict[str, Any]:
        return _mt5_to_dict(self.mt5.symbol_info_tick(symbol))

    async def create_market_buy_order(self, symbol: str, volume: float, sl: float, tp: float, deviation: int = 20, magic: int = 0, comment: str | None = None) -> dict[str, Any]:
        price = self.mt5.symbol_info_tick(symbol).ask
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": self.mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": comment or "forex_bot",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        return _mt5_to_dict(result)

    async def create_market_sell_order(self, symbol: str, volume: float, sl: float, tp: float, deviation: int = 20, magic: int = 0, comment: str | None = None) -> dict[str, Any]:
        price = self.mt5.symbol_info_tick(symbol).bid
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": comment or "forex_bot",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        return _mt5_to_dict(result)

    async def close_position_by_ticket(self, ticket: int, symbol: str, volume: float | None = None) -> dict[str, Any]:
        positions = self.mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "reason": "Position not found"}
        pos = positions[0]
        pos_dict = _mt5_to_dict(pos)
        vol = float(volume if volume is not None else pos_dict.get("volume", 0.0))
        order_type = self.mt5.ORDER_TYPE_SELL if pos_dict.get("type", 0) == 0 else self.mt5.ORDER_TYPE_BUY
        price = self.mt5.symbol_info_tick(symbol).bid if order_type == self.mt5.ORDER_TYPE_SELL else self.mt5.symbol_info_tick(symbol).ask
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": vol,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": pos_dict.get("magic", 0),
            "comment": "forex_bot_close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        return _mt5_to_dict(result)


class MT5Adapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.connection = _StubMT5Connection()
        self._connected = False

    async def connect(self) -> bool:
        backend = str(getattr(self.settings.system, "broker_backend", ""))
        if backend != "mt5_local":
            self._connected = False
            return False
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:
            logger.warning("MetaTrader5 not available: %s", exc)
            self._connected = False
            return False
        try:
            if not mt5.initialize():
                self._connected = False
                return False
            self._connected = True
            self.connection = _MT5Connection(mt5)
        except Exception as exc:
            logger.warning("MT5 initialize failed: %s", exc)
            self._connected = False
            return False
        return True

    async def disconnect(self) -> None:
        try:
            import MetaTrader5 as mt5  # type: ignore

            mt5.shutdown()
        except Exception:
            pass
        self._connected = False
        self.connection = _StubMT5Connection()

    def is_connected(self) -> bool:
        return bool(self._connected)

    async def get_live_frames(self, symbol: str, *, timeframes: list[str], tail: int = 500) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        if not self._connected:
            return frames
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception:
            return frames
        tf_const_name = {
            "M1": "TIMEFRAME_M1",
            "M2": "TIMEFRAME_M2",
            "M3": "TIMEFRAME_M3",
            "M4": "TIMEFRAME_M4",
            "M5": "TIMEFRAME_M5",
            "M6": "TIMEFRAME_M6",
            "M10": "TIMEFRAME_M10",
            "M12": "TIMEFRAME_M12",
            "M15": "TIMEFRAME_M15",
            "M20": "TIMEFRAME_M20",
            "M30": "TIMEFRAME_M30",
            "H1": "TIMEFRAME_H1",
            "H2": "TIMEFRAME_H2",
            "H3": "TIMEFRAME_H3",
            "H4": "TIMEFRAME_H4",
            "H6": "TIMEFRAME_H6",
            "H8": "TIMEFRAME_H8",
            "H12": "TIMEFRAME_H12",
            "D1": "TIMEFRAME_D1",
            "W1": "TIMEFRAME_W1",
            "MN1": "TIMEFRAME_MN1",
        }
        tf_map = {tf: getattr(mt5, const, None) for tf, const in tf_const_name.items()}
        for tf in timeframes:
            mt5_tf = tf_map.get(tf)
            if mt5_tf is None:
                continue
            try:
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, tail)
            except Exception:
                rates = None
            if rates is None or len(rates) == 0:
                continue
            df = pd.DataFrame(rates)
            df = _normalize_columns(df)
            df = _ensure_datetime_index(df)
            try:
                df.attrs["source"] = "mt5"
            except Exception:
                pass
            frames[tf] = df
        # Fill unavailable MT5 timeframes via deterministic resampling from available data.
        frames = _resample_missing_from_available(frames, [str(tf).upper() for tf in timeframes])
        return frames


class DataLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = Path(getattr(self.settings.system, "data_dir", "data"))
        self.mt5_adapter = MT5Adapter(settings)
        self._connected = False

    async def connect(self) -> bool:
        self._connected = await self.mt5_adapter.connect()
        return self._connected

    async def disconnect(self) -> None:
        await self.mt5_adapter.disconnect()
        self._connected = False

    def is_connected(self) -> bool:
        return bool(self._connected)

    def _timeframes(self) -> list[str]:
        base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1")
        tfs = [base_tf]
        use_multires = bool(getattr(self.settings.system, "multi_resolution_enabled", True))
        if use_multires:
            for tf in list(getattr(self.settings.system, "multi_resolution_timeframes", []) or []):
                tfu = str(tf or "").upper()
                if tfu and tfu not in tfs:
                    tfs.append(tfu)
        for tf in list(getattr(self.settings.system, "required_timeframes", []) or []):
            tfu = str(tf or "").upper()
            if tfu and tfu not in tfs:
                tfs.append(tfu)
        for tf in list(getattr(self.settings.system, "higher_timeframes", []) or []):
            tfu = str(tf or "").upper()
            if tfu and tfu not in tfs:
                tfs.append(tfu)
        return _ordered_timeframes(tfs)

    def _candidate_paths(self, symbol: str, tf: str) -> list[Path]:
        paths = []
        root = self.data_dir
        paths.append(root / f"symbol={symbol}" / f"timeframe={tf}" / "data.parquet")
        paths.append(root / f"symbol={symbol}" / f"timeframe={tf}" / "data.csv")
        paths.append(root / f"{symbol}_{tf}.parquet")
        paths.append(root / f"{symbol}_{tf}.csv")
        paths.append(root / symbol / f"{tf}.parquet")
        paths.append(root / symbol / f"{tf}.csv")
        return paths

    def _load_frames(self, symbol: str, *, tail: int | None = None, allow_resample: bool = True) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        if _use_rust_data_backend():
            try:
                rust_frames = self._load_frames_rust(symbol, tail=tail, allow_resample=allow_resample)
                if rust_frames:
                    return rust_frames
            except Exception as exc:
                _disable_rust_data_backend()
                logger.warning("Rust data loader failed; falling back to Python: %s", exc)
        if not self.data_dir.exists():
            return frames
        for tf in self._timeframes():
            df = None
            for path in self._candidate_paths(symbol, tf):
                df = _read_frame(path)
                if df is not None and not df.empty:
                    break
            if df is None or df.empty:
                continue
            df = _normalize_columns(df)
            df = _ensure_datetime_index(df)
            df = _ensure_ohlcv(df)
            try:
                df.attrs["source"] = "disk"
            except Exception:
                pass
            if tail is not None and tail > 0:
                df = df.tail(tail)
            frames[tf] = df
        if allow_resample:
            frames = _resample_missing_from_available(frames, self._timeframes())
        return frames

    def _load_frames_rust(
        self,
        symbol: str,
        *,
        tail: int | None = None,
        allow_resample: bool = True,
    ) -> dict[str, pd.DataFrame]:
        try:
            import forex_bindings  # type: ignore
        except Exception:
            _disable_rust_data_backend()
            return {}
        if not hasattr(forex_bindings, "load_symbol_frames"):
            _disable_rust_data_backend()
            return {}

        if not self.data_dir.exists():
            return {}

        base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1")
        tfs = self._timeframes()

        payload = forex_bindings.load_symbol_frames(
            root=str(self.data_dir),
            symbol=symbol,
            timeframes=tfs,
            resample_missing=bool(allow_resample),
            base_tf=base_tf,
        )
        if not isinstance(payload, dict):
            return {}

        frames: dict[str, pd.DataFrame] = {}
        for tf, frame in payload.items():
            if not isinstance(frame, dict):
                continue
            try:
                data = {k: np.asarray(v) for k, v in frame.items()}
                df = pd.DataFrame(data)
            except Exception:
                continue
            df = _normalize_columns(df)
            df = _ensure_datetime_index(df)
            df = _ensure_ohlcv(df)
            try:
                df.attrs["source"] = "disk"
            except Exception:
                pass
            if tail is not None and tail > 0:
                df = df.tail(tail)
            frames[str(tf)] = df
        return frames

    async def ensure_history(self, symbol: str) -> bool:
        frames = self._load_frames(symbol)
        return bool(frames)

    async def ensure_all_history(self, symbols: list[str]) -> bool:
        if not self.data_dir.exists():
            logger.warning(
                "Data directory '%s' not found. Expected layout: "
                "data/symbol=EURUSD/timeframe=M1/data.parquet (or data/EURUSD_M1.csv).",
                self.data_dir,
            )
            return False
        any_found = False
        for sym in symbols:
            try:
                if await self.ensure_history(sym):
                    any_found = True
                else:
                    logger.warning("No local history found for symbol %s.", sym)
            except Exception as exc:
                logger.warning("History check failed for %s: %s", sym, exc)
        return any_found

    async def get_training_data(self, symbol: str) -> dict[str, pd.DataFrame]:
        return self._load_frames(symbol)

    async def get_live_data(self, symbol: str) -> dict[str, pd.DataFrame]:
        tfs = self._timeframes()
        if self.mt5_adapter.is_connected():
            frames = await self.mt5_adapter.get_live_frames(symbol, timeframes=tfs, tail=500)
            if frames:
                frames = _resample_missing_from_available(frames, tfs)
                return frames
        return self._load_frames(symbol, tail=2000)


__all__ = ["DataLoader"]
