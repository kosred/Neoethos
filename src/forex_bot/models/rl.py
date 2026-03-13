from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import psutil
except ImportError:
    psutil = None

from ..core.system import resolve_cpu_budget

# Optional Stable Baselines 3
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        EvalCallback,
        StopTrainingOnNoModelImprovement,
    )

    SB3_AVAILABLE = True
except ImportError:
    PPO = None
    SAC = None
    DummyVecEnv = None
    SubprocVecEnv = None
    BaseCallback = None
    CallbackList = None
    EvalCallback = None
    StopTrainingOnNoModelImprovement = None
    SB3_AVAILABLE = False

from .base import (
    ExpertModel,
    get_early_stop_params,
    time_series_train_val_split,
    validate_time_ordering,
)

logger = logging.getLogger(__name__)

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _update_state_numba(
        action, position, entry_price, current_close, current_high, current_low, 
        balance, equity, high_water_mark, daily_start_equity, commission, 
        initial_balance, profit_target, max_daily_dd, max_total_dd
    ):
        prev_equity = equity
        new_position = position
        new_entry_price = entry_price
        new_balance = balance
        
        # 1. Action Logic (Standard)
        if action == 1: # Buy
            if position == -1: 
                new_balance = equity
                new_position = 0
            if new_position == 0: 
                new_position = 1
                new_entry_price = current_close
                new_balance -= new_balance * commission
                equity = new_balance
        elif action == 2: # Sell
            if position == 1: 
                new_balance = equity
                new_position = 0
            if new_position == 0: 
                new_position = -1
                new_entry_price = current_close
                new_balance -= new_balance * commission
                equity = new_balance

        # 2. Real-time High/Low Equity Tracking (HPC FIX)
        # We check the WORST case scenario during this bar to catch DD breaches
        worst_floating_equity = equity
        if new_position == 1: # Long
            worst_floating_equity = new_balance * (1.0 + (current_low - new_entry_price) / new_entry_price)
            equity = new_balance * (1.0 + (current_close - new_entry_price) / new_entry_price)
        elif new_position == -1: # Short
            worst_floating_equity = new_balance * (1.0 + (new_entry_price - current_high) / new_entry_price)
            equity = new_balance * (1.0 + (new_entry_price - current_close) / new_entry_price)
        
        new_high_water = max(high_water_mark, equity)
        
        # 3. Reward & Done Logic (Using WORST equity for DD checks)
        pnl_change = (equity - prev_equity) / prev_equity
        reward = pnl_change * 100.0
        
        # Prop-firm DISQUALIFICATION check
        daily_dd_worst = (daily_start_equity - worst_floating_equity) / daily_start_equity
        total_dd_worst = (new_high_water - worst_floating_equity) / new_high_water
        
        done = False
        if daily_dd_worst >= max_daily_dd or total_dd_worst >= max_total_dd:
            reward = -150.0 # Heavier penalty for account death
            done = True
            
        current_return = (equity - initial_balance) / initial_balance
        if current_return >= profit_target and (prev_equity - initial_balance)/initial_balance < profit_target:
            reward += 50.0
            
        if new_position != 0: reward -= 0.001
        
        return new_position, new_entry_price, new_balance, equity, new_high_water, reward, done
else:
    def _update_state_numba(
        action, position, entry_price, current_close, current_high, current_low, 
        balance, equity, high_water_mark, daily_start_equity, commission, 
        initial_balance, profit_target, max_daily_dd, max_total_dd
    ):
        prev_equity = equity
        new_position = position
        new_entry_price = entry_price
        new_balance = balance
        
        # 1. Action Logic (Standard)
        if action == 1: # Buy
            if position == -1: 
                new_balance = equity
                new_position = 0
            if new_position == 0: 
                new_position = 1
                new_entry_price = current_close
                new_balance -= new_balance * commission
                equity = new_balance
        elif action == 2: # Sell
            if position == 1: 
                new_balance = equity
                new_position = 0
            if new_position == 0: 
                new_position = -1
                new_entry_price = current_close
                new_balance -= new_balance * commission
                equity = new_balance

        # 2. Real-time High/Low Equity Tracking
        worst_floating_equity = equity
        if new_position == 1: # Long
            worst_floating_equity = new_balance * (1.0 + (current_low - new_entry_price) / new_entry_price)
            equity = new_balance * (1.0 + (current_close - new_entry_price) / new_entry_price)
        elif new_position == -1: # Short
            worst_floating_equity = new_balance * (1.0 + (new_entry_price - current_high) / new_entry_price)
            equity = new_balance * (1.0 + (new_entry_price - current_close) / new_entry_price)
        
        new_high_water = max(high_water_mark, equity)
        
        # 3. Reward & Done Logic
        pnl_change = (equity - prev_equity) / prev_equity
        reward = pnl_change * 100.0
        
        # Prop-firm DISQUALIFICATION check
        daily_dd_worst = (daily_start_equity - worst_floating_equity) / daily_start_equity
        total_dd_worst = (new_high_water - worst_floating_equity) / new_high_water
        
        done = False
        if daily_dd_worst >= max_daily_dd or total_dd_worst >= max_total_dd:
            reward = -150.0 
            done = True
            
        current_return = (equity - initial_balance) / initial_balance
        if current_return >= profit_target and (prev_equity - initial_balance)/initial_balance < profit_target:
            reward += 50.0
            
        if new_position != 0: reward -= 0.001
        
        return new_position, new_entry_price, new_balance, equity, new_high_water, reward, done


def _is_dataframe_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "index"))


def _is_frame_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "__getitem__"))


def _row_count(value: Any) -> int:
    if value is None:
        return 0
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and len(shape) > 0:
        try:
            return int(shape[0])
        except Exception:
            pass
    if isinstance(value, dict):
        for v in value.values():
            try:
                return int(np.asarray(v).reshape(-1).shape[0])
            except Exception:
                continue
        return 0
    try:
        return int(len(value))
    except Exception:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return 1
        return int(arr.shape[0])


def _slice_by_indices(obj: Any, indices: Any) -> Any:
    if obj is None:
        return None
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if _is_dataframe_like(obj):
        try:
            return obj.take(idx)
        except Exception:
            try:
                base_idx = np.asarray(getattr(obj, "index")).reshape(-1)
                return obj.loc[base_idx[idx]]
            except Exception:
                pass
    if _is_frame_like(obj):
        cols = getattr(obj, "columns", None)
        names: list[str] = []
        if cols is not None:
            try:
                names = [str(c) for c in list(cols)]
            except Exception:
                names = []
        out: dict[str, Any] = {}
        for col in names:
            try:
                vec = np.asarray(obj[col]).reshape(-1)  # type: ignore[index]
                out[col] = vec[idx]
            except Exception:
                continue
        src_idx = getattr(obj, "index", None)
        if src_idx is not None:
            src_arr = np.asarray(src_idx).reshape(-1)
            out["index"] = src_arr[idx]
        return out
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return arr
    return arr[idx]


def _slice_rows(obj: Any, start: int, end: int | None = None) -> Any:
    if obj is None:
        return None
    s = max(0, int(start))
    if _is_dataframe_like(obj) or _is_frame_like(obj):
        n = _row_count(obj)
        e = n if end is None else min(n, max(s, int(end)))
        return _slice_by_indices(obj, np.arange(s, e, dtype=np.int64))
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return arr
    if end is None:
        return arr[s:]
    return arr[s:end]


def _tail_rows(obj: Any, max_rows: int) -> Any:
    n = _row_count(obj)
    keep = max(0, int(max_rows))
    if keep <= 0 or n <= keep:
        return obj
    return _slice_rows(obj, n - keep, None)


def _to_numpy_1d(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        if dtype is None:
            arr = np.asarray(values.to_numpy(copy=False))
        else:
            arr = np.asarray(values.to_numpy(dtype=dtype, copy=False), dtype=dtype)
    else:
        arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        return np.asarray([arr.item()], dtype=arr.dtype)
    return arr.reshape(-1)


def _to_numpy_2d_float32(values: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        arr = np.asarray(values.to_numpy(dtype=np.float32, copy=False), dtype=np.float32)
    else:
        arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


def _column_to_numpy(df: Any, column: str, *, dtype: Any | None = None) -> np.ndarray:
    if not hasattr(df, "__getitem__"):
        raise ValueError(f"Input does not support column access for {column!r}.")
    try:
        col = df[column]  # type: ignore[index]
    except Exception as exc:
        raise KeyError(f"Missing required column {column!r}") from exc
    return _to_numpy_1d(col, dtype=dtype)


def _day_of_month(values: Any, *, n_rows: int) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return np.ones(n_rows, dtype=np.int32)
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            day = (arr.astype("datetime64[D]") - arr.astype("datetime64[M]")).astype(np.int32) + 1
            return day.astype(np.int32, copy=False)
    except Exception:
        pass
    out = np.ones(arr.shape[0], dtype=np.int32)
    for i, val in enumerate(arr):
        d = getattr(val, "day", None)
        if d is not None:
            try:
                out[i] = int(d)
            except Exception:
                out[i] = 1
    return out

class PropFirmTradingEnv(gym.Env):
    """
    A specialized Reinforcement Learning environment for Prop Firm challenges.

    Objectives:
    1. Consistency: Target ~4% monthly return.
    2. Survival: Strictly penalize hitting 4% Daily DD or 8% Max DD.
    3. Sortino: Penalize downside volatility.

    Observation Space:
    - Market Features (from FeatureEngineer)
    - Account State (Daily DD, Total DD, Current Profit)

    Action Space:
    - Discrete: 0=Hold, 1=Buy, 2=Sell
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: Any,
        features: Any,
        initial_balance: float = 100000.0,
        max_daily_dd: float = 0.04,
        max_total_dd: float = 0.08,
        profit_target: float = 0.04,
        commission_pct: float = 0.00005,  # ~0.5 pips
    ):
        super().__init__()

        if df is None or len(df) == 0:
            raise ValueError("PropFirmTradingEnv requires a non-empty DataFrame.")
        if features is None or len(features) == 0:
            raise ValueError("PropFirmTradingEnv requires non-empty feature data.")
        if len(df) != len(features):
            logger.warning(
                "PropFirmTradingEnv data length mismatch: df=%s features=%s",
                len(df),
                len(features),
            )

        self.df = df
        self.features = _to_numpy_2d_float32(features)
        # HPC: Vectorized price access
        self.prices = _column_to_numpy(df, "close", dtype=np.float32)
        self.highs = _column_to_numpy(df, "high", dtype=np.float32)
        self.lows = _column_to_numpy(df, "low", dtype=np.float32)
        
        # Track days vectorized for speed
        if hasattr(df, "columns") and "timestamp" in df.columns:
            ts = _column_to_numpy(df, "timestamp")
            self.day_indices = _day_of_month(ts, n_rows=len(df))
        else:
            idx = getattr(df, "index", None)
            self.day_indices = _day_of_month(idx if idx is not None else np.arange(len(df)), n_rows=len(df))
        if self.day_indices.shape[0] != len(df):
            self.day_indices = np.ones(len(df), dtype=np.int32)
            
        self.n_steps = len(df)
        self.initial_balance = initial_balance
        self.max_daily_dd = max_daily_dd
        self.max_total_dd = max_total_dd
        self.profit_target = profit_target
        self.commission = commission_pct

        # State
        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.high_water_mark = initial_balance
        self.daily_start_equity = initial_balance
        self.position = 0  # 0=Flat, 1=Long, -1=Short
        self.entry_price = 0.0
        self.current_day_idx = 0

        # Action: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation: [Features... , Daily_DD_Pct, Total_DD_Pct, PnL_Pct]
        n_features = self.features.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features + 3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.high_water_mark = self.initial_balance
        self.daily_start_equity = self.initial_balance
        self.position = 0
        self.entry_price = 0.0

        # Determine day index for daily reset
        self.current_day_idx = int(self.day_indices[0]) if self.day_indices.size > 0 else 1

        return self._get_obs(), {}

    def _get_obs(self):
        # Market State
        feat = self.features[self.current_step]

        # Account State
        daily_dd = (self.daily_start_equity - self.equity) / self.daily_start_equity
        total_dd = (self.high_water_mark - self.equity) / self.high_water_mark
        pnl_pct = (self.equity - self.initial_balance) / self.initial_balance

        state = np.concatenate([feat, [daily_dd, total_dd, pnl_pct]])
        return state.astype(np.float32)

    def step(self, action):
        current_price = self.prices[self.current_step]
        current_high = self.highs[self.current_step]
        current_low = self.lows[self.current_step]
        
        # HPC: Execute the entire market simulation in machine code
        (
            self.position, 
            self.entry_price, 
            self.balance, 
            self.equity, 
            self.high_water_mark, 
            step_reward, 
            done
        ) = _update_state_numba(
            int(action), self.position, self.entry_price, 
            float(current_price), float(current_high), float(current_low),
            self.balance, self.equity, self.high_water_mark, self.daily_start_equity, 
            self.commission, self.initial_balance, self.profit_target, 
            self.max_daily_dd, self.max_total_dd
        )

        # 3. Time & Day Handling
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        # Check Daily Reset
        if not done:
            # HPC: Fast day lookup
            current_day = self.day_indices[self.current_step]
            if current_day != self.current_day_idx:
                self.current_day_idx = current_day
                self.daily_start_equity = self.equity

        return self._get_obs(), float(step_reward), done, done, {}

    def _open_position(self, price, direction):
        self.position = direction
        self.entry_price = price
        # Pay commission immediately
        self.equity -= self.equity * self.commission
        self.balance -= self.balance * self.commission

    def _close_position(self, price):
        # PnL logic handled in continuous update, just reset flags
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.equity  # Realize PnL

    def _update_equity(self, price):
        if self.position == 0:
            return

        delta = (price - self.entry_price) / self.entry_price
        if self.position == -1:
            delta = -delta

        # Unrealized PnL applied to Balance
        # Note: In strict accounting, Equity = Balance + Unrealized.
        # Here we simplify: self.balance is "Cash at Entry", self.equity is floating.
        self.equity = self.balance * (1.0 + delta)

        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity


if BaseCallback is not None:
    class _TimeLimitCallback(BaseCallback):
        def __init__(self, max_time_sec: int, verbose: int = 0) -> None:
            super().__init__(verbose)
            self.max_time_sec = max(0, int(max_time_sec))
            self._start_time: float | None = None

        def _on_training_start(self) -> None:
            self._start_time = time.time()

        def _on_step(self) -> bool:
            if self.max_time_sec <= 0 or self._start_time is None:
                return True
            if time.time() - self._start_time > self.max_time_sec:
                if self.model is not None:
                    self.model.stop_training = True
                return False
            return True
else:
    _TimeLimitCallback = None


class RLExpertPPO(ExpertModel):
    """PPO Agent wrapper compatible with ForexBot ensemble."""

    def __init__(
        self,
        timesteps: int = 1_000_000,
        max_time_sec: int = 1800,
        device: str = "auto",
        network_arch: list[int] | None = None,
        eval_ratio: float = 0.15,
        eval_max_rows: int = 200_000,
        eval_freq: int = 0,
        parallel_envs: int = 1,
        **kwargs,
    ):
        self.model = None
        self.env = None
        self.timesteps = int(timesteps)
        self.max_time_sec = int(max_time_sec) if max_time_sec else 0
        self.device = device
        self.network_arch = network_arch
        self.eval_ratio = float(eval_ratio)
        self.eval_max_rows = int(eval_max_rows) if eval_max_rows else 0
        self.eval_freq = int(eval_freq) if eval_freq else 0
        self.parallel_envs = int(parallel_envs) if parallel_envs else 1

    @staticmethod
    def _resolve_device(device: str) -> str:
        if not isinstance(device, str) or device.strip() == "" or device == "auto":
            return "auto"
        if device.startswith("cuda"):
            return "cuda"
        return "cpu"

    def _split_env_data(
        self,
        X: Any,
        y: Any,
        metadata: Any,
    ) -> tuple[Any, Any, Any | None, Any | None]:
        try:
            X_train, X_val, _, _ = time_series_train_val_split(
                X, y, val_ratio=self.eval_ratio, min_train_samples=100, embargo_samples=0
            )
        except Exception:
            split = int(len(X) * (1.0 - self.eval_ratio))
            X_train = _slice_rows(X, 0, split)
            X_val = _slice_rows(X, split, None)

        meta_train = None
        meta_val = None
        if metadata is not None:
            try:
                # Align metadata with the split data using index
                if _row_count(X_train) > 0 and hasattr(metadata, "loc") and hasattr(X_train, "index"):
                    meta_train = metadata.loc[X_train.index]
                if _row_count(X_val) > 0 and hasattr(metadata, "loc") and hasattr(X_val, "index"):
                    meta_val = metadata.loc[X_val.index]
            except Exception:
                # Fallback: positional slicing if index alignment fails
                split_idx = _row_count(X_train)
                meta_train = _slice_rows(metadata, 0, split_idx) if split_idx > 0 else None
                meta_val = _slice_rows(metadata, split_idx, None) if _row_count(X_val) > 0 else None
        return X_train, X_val, meta_train, meta_val

    def fit(self, X: Any, y: Any, **kwargs):
        if not SB3_AVAILABLE:
            logger.warning("StableBaselines3 not installed. RL skipped.")
            return

        # Validate time-ordering to prevent look-ahead bias
        validate_time_ordering(X, context="RLExpertPPO.fit")

        metadata = kwargs.get("metadata")
        if metadata is None:
            logger.warning("RL Expert requires metadata (OHLC) for Prop Environment.")
            return

        X_train, X_val, meta_train, meta_val = self._split_env_data(X, y, metadata)
        if meta_train is None or _row_count(X_train) == 0:
            logger.warning("RL Expert: training metadata unavailable or empty.")
            return

        if meta_val is not None and self.eval_max_rows > 0 and _row_count(X_val) > self.eval_max_rows:
            X_val = _tail_rows(X_val, self.eval_max_rows)
            meta_val = _tail_rows(meta_val, self.eval_max_rows)
        x_train_arr = _to_numpy_2d_float32(X_train)
        x_val_arr = _to_numpy_2d_float32(X_val) if _row_count(X_val) > 0 else np.zeros((0, 0), dtype=np.float32)

        # Create environment with high-speed parallel execution
        import os
        cpu_budget = resolve_cpu_budget()
        try:
            requested = int(os.environ.get("FOREX_BOT_RL_ENVS", cpu_budget) or cpu_budget)
        except Exception:
            requested = cpu_budget
        n_envs = max(1, min(requested, cpu_budget))

        # Use default arguments to capture closure variables at function creation time
        def make_env(m=meta_train, x=x_train_arr):
            return PropFirmTradingEnv(m, x)

        # RAM-aware parallelization: Check if we have memory for parallel envs
        if n_envs > 1 and psutil:
            available_gb = psutil.virtual_memory().available / 1e9
            data_size_gb = x_train_arr.nbytes / 1e9
            # Each SubprocVecEnv process needs a copy of data (2x safety margin)
            affordable_envs = max(1, int(available_gb / (data_size_gb * 2.0)))
            if n_envs > affordable_envs:
                logger.info(f"RAM-limited: reducing RL envs from {n_envs} to {affordable_envs} "
                           f"(available: {available_gb:.1f}GB, data: {data_size_gb:.1f}GB per env)")
                n_envs = affordable_envs

        # Python 3.13+ on Windows: SubprocVecEnv may fail with pickle errors
        # Fallback to DummyVecEnv (single process) if multiprocessing fails
        if n_envs > 1:
            try:
                self.env = SubprocVecEnv([make_env for _ in range(n_envs)])
            except (OSError, EOFError, BrokenPipeError, MemoryError, Exception) as e:
                logger.warning(f"SubprocVecEnv failed ({e.__class__.__name__}), falling back to single-process DummyVecEnv")
                self.env = DummyVecEnv([make_env])
        else:
            self.env = DummyVecEnv([make_env])

        env_is_subproc = SubprocVecEnv is not None and isinstance(self.env, SubprocVecEnv)
        eval_env = None
        if meta_val is not None and x_val_arr.shape[0] > 0 and EvalCallback is not None:
            # Use default arguments to capture closure variables (named function for pickling)
            def make_eval_env(m=meta_val, x=x_val_arr):
                return PropFirmTradingEnv(m, x)
            if env_is_subproc:
                try:
                    eval_env = SubprocVecEnv([make_eval_env])
                except (OSError, EOFError, BrokenPipeError, MemoryError, Exception) as e:
                    logger.warning(
                        f"SubprocVecEnv eval failed ({e.__class__.__name__}), using DummyVecEnv"
                    )
                    eval_env = DummyVecEnv([make_eval_env])
            else:
                eval_env = DummyVecEnv([make_eval_env])

        # Larger policy on GPU; smaller on CPU
        policy_kwargs = {}
        try:
            import torch

            is_cuda = torch.cuda.is_available() and self._resolve_device(self.device) != "cpu"
        except Exception:
            is_cuda = False
        if self.network_arch:
            policy_kwargs["net_arch"] = list(self.network_arch)
        else:
            policy_kwargs["net_arch"] = [512, 512, 512] if is_cuda else [256, 256]

        sb3_device = self._resolve_device(self.device)
        if sb3_device == "auto" and not is_cuda:
            sb3_device = "cpu"

        try:
            self.model = PPO("MlpPolicy", self.env, verbose=0, device=sb3_device, policy_kwargs=policy_kwargs)

            callbacks = []
            if _TimeLimitCallback is not None and self.max_time_sec > 0:
                callbacks.append(_TimeLimitCallback(self.max_time_sec))
            if (
                eval_env is not None
                and EvalCallback is not None
                and StopTrainingOnNoModelImprovement is not None
            ):
                patience, _ = get_early_stop_params(5, 0.0)
                eval_freq = self.eval_freq or max(1000, int(self.timesteps // 10))
                stop_cb = StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=patience,
                    min_evals=3,
                    verbose=0,
                )
                eval_cb = EvalCallback(
                    eval_env,
                    eval_freq=eval_freq,
                    n_eval_episodes=1,
                    deterministic=True,
                    callback_after_eval=stop_cb,
                )
                callbacks.append(eval_cb)

            callback = CallbackList(callbacks) if callbacks and CallbackList is not None else None
            total_timesteps = max(1, int(self.timesteps))
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
        except Exception as exc:
            logger.warning("RLExpertPPO training skipped due to runtime error: %s", exc)
            self.model = None
            return

    def predict_proba(self, X: Any, **kwargs) -> np.ndarray:
        if self.model is None:
            return np.zeros((_row_count(X), 3))

        # This is tricky: RL agents are sequential. Predicting a batch 'offline'
        # requires simulating the env state for each row.
        # For ensemble compatibility, we run a deterministic pass.
        x_arr = _to_numpy_2d_float32(X)
        actions, _ = self.model.predict(x_arr, deterministic=True)

        # Map actions (0,1,2) to probas (one-hot)
        probs = np.zeros((len(actions), 3))
        for i, a in enumerate(actions):
            probs[i, a] = 1.0
        return probs

    def save(self, path):
        if self.model:
            self.model.save(f"{path}/rl_ppo.zip")

    def load(self, path):
        if SB3_AVAILABLE:
            try:
                self.model = PPO.load(f"{path}/rl_ppo.zip")
            except Exception:
                pass


class RLExpertSAC(RLExpertPPO):
    """SAC Agent wrapper."""

    def fit(self, X: Any, y: Any, **kwargs):
        if not SB3_AVAILABLE:
            logger.warning("StableBaselines3 not installed. RL skipped.")
            return

        # Validate time-ordering to prevent look-ahead bias
        validate_time_ordering(X, context="RLExpertSAC.fit")

        metadata = kwargs.get("metadata")
        if metadata is None:
            logger.warning("RL Expert requires metadata (OHLC) for Prop Environment.")
            return

        X_train, X_val, meta_train, meta_val = self._split_env_data(X, y, metadata)
        if meta_train is None or _row_count(X_train) == 0:
            logger.warning("RL Expert: training metadata unavailable or empty.")
            return

        if meta_val is not None and self.eval_max_rows > 0 and _row_count(X_val) > self.eval_max_rows:
            X_val = _tail_rows(X_val, self.eval_max_rows)
            meta_val = _tail_rows(meta_val, self.eval_max_rows)
        x_train_arr = _to_numpy_2d_float32(X_train)
        x_val_arr = _to_numpy_2d_float32(X_val) if _row_count(X_val) > 0 else np.zeros((0, 0), dtype=np.float32)

        # Create environment with high-speed parallel execution
        import os
        cpu_budget = resolve_cpu_budget()
        try:
            requested = int(os.environ.get("FOREX_BOT_RL_ENVS", cpu_budget) or cpu_budget)
        except Exception:
            requested = cpu_budget
        n_envs = max(1, min(requested, cpu_budget))

        # Use default arguments to capture closure variables at function creation time
        def make_env(m=meta_train, x=x_train_arr):
            return PropFirmTradingEnv(m, x)

        # RAM-aware parallelization: Check if we have memory for parallel envs
        if n_envs > 1 and psutil:
            available_gb = psutil.virtual_memory().available / 1e9
            data_size_gb = x_train_arr.nbytes / 1e9
            # Each SubprocVecEnv process needs a copy of data (2x safety margin)
            affordable_envs = max(1, int(available_gb / (data_size_gb * 2.0)))
            if n_envs > affordable_envs:
                logger.info(f"RAM-limited: reducing RL envs from {n_envs} to {affordable_envs} "
                           f"(available: {available_gb:.1f}GB, data: {data_size_gb:.1f}GB per env)")
                n_envs = affordable_envs

        # Python 3.13+ on Windows: SubprocVecEnv may fail with pickle errors
        # Fallback to DummyVecEnv (single process) if multiprocessing fails
        if n_envs > 1:
            try:
                self.env = SubprocVecEnv([make_env for _ in range(n_envs)])
            except (OSError, EOFError, BrokenPipeError, MemoryError, Exception) as e:
                logger.warning(f"SubprocVecEnv failed ({e.__class__.__name__}), falling back to single-process DummyVecEnv")
                self.env = DummyVecEnv([make_env])
        else:
            self.env = DummyVecEnv([make_env])

        env_is_subproc = SubprocVecEnv is not None and isinstance(self.env, SubprocVecEnv)
        eval_env = None
        if meta_val is not None and x_val_arr.shape[0] > 0 and EvalCallback is not None:
            # Use default arguments to capture closure variables (named function for pickling)
            def make_eval_env(m=meta_val, x=x_val_arr):
                return PropFirmTradingEnv(m, x)
            if env_is_subproc:
                try:
                    eval_env = SubprocVecEnv([make_eval_env])
                except (OSError, EOFError, BrokenPipeError, MemoryError, Exception) as e:
                    logger.warning(
                        f"SubprocVecEnv eval failed ({e.__class__.__name__}), using DummyVecEnv"
                    )
                    eval_env = DummyVecEnv([make_eval_env])
            else:
                eval_env = DummyVecEnv([make_eval_env])

        # Larger policy on GPU; smaller on CPU
        policy_kwargs = {}
        try:
            import torch

            is_cuda = torch.cuda.is_available() and self._resolve_device(self.device) != "cpu"
        except Exception:
            is_cuda = False
        if self.network_arch:
            policy_kwargs["net_arch"] = list(self.network_arch)
        else:
            policy_kwargs["net_arch"] = [512, 512, 512] if is_cuda else [256, 256]

        sb3_device = self._resolve_device(self.device)
        if sb3_device == "auto" and not is_cuda:
            sb3_device = "cpu"

        if not isinstance(getattr(self.env, "action_space", None), spaces.Box):
            logger.warning(
                "RLExpertSAC requires continuous action space; got %s. Skipping.",
                type(getattr(self.env, "action_space", None)).__name__,
            )
            self.model = None
            return

        try:
            self.model = SAC("MlpPolicy", self.env, verbose=0, device=sb3_device, policy_kwargs=policy_kwargs)

            callbacks = []
            if _TimeLimitCallback is not None and self.max_time_sec > 0:
                callbacks.append(_TimeLimitCallback(self.max_time_sec))
            if (
                eval_env is not None
                and EvalCallback is not None
                and StopTrainingOnNoModelImprovement is not None
            ):
                patience, _ = get_early_stop_params(5, 0.0)
                eval_freq = self.eval_freq or max(1000, int(self.timesteps // 10))
                stop_cb = StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=patience,
                    min_evals=3,
                    verbose=0,
                )
                eval_cb = EvalCallback(
                    eval_env,
                    eval_freq=eval_freq,
                    n_eval_episodes=1,
                    deterministic=True,
                    callback_after_eval=stop_cb,
                )
                callbacks.append(eval_cb)

            callback = CallbackList(callbacks) if callbacks and CallbackList is not None else None
            total_timesteps = max(1, int(self.timesteps))
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
        except Exception as exc:
            logger.warning("RLExpertSAC training skipped due to runtime error: %s", exc)
            self.model = None
            return

    def save(self, path):
        if self.model:
            self.model.save(f"{path}/rl_sac.zip")

    def load(self, path):
        if SB3_AVAILABLE:
            try:
                self.model = SAC.load(f"{path}/rl_sac.zip")
            except Exception:
                pass


def _build_continuous_env(df: Any, y: Any) -> gym.Env:
    """Helper to build env for optimization (placeholder)."""
    return PropFirmTradingEnv(df, _to_numpy_2d_float32(df))



