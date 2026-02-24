import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..models.exit_agent import ExitAgent
from .mt5_state_manager import MT5Position

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CloseInstruction:
    ticket: int
    symbol: str
    volume: float  # 0.0 means close all
    reason: str
    score: float


try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@njit(cache=True, fastmath=True)
def _diagnose_positions_numba(
    pos_types, pos_prices_open, pos_sls, pos_durations,
    current_price, momentum, rsi, volatility,
    stall_threshold, min_profit_bank, reversal_sens, zombie_minutes
):
    n = len(pos_types)
    results = np.zeros(n, dtype=np.int8) # 0=Hold, 1=Close
    
    for i in range(n):
        p_type = pos_types[i]
        p_open = pos_prices_open[i]
        p_dur = pos_durations[i]
        
        # HPC FIX: Volatility-Normalized Distance
        # We use ATR (volatility) as the yardstick instead of hard SL
        # Standard SL is usually ~2 ATR
        effective_risk = max(volatility * 2.0, 1e-9)
        
        curr_dist = (current_price - p_open) if p_type == 0 else (p_open - current_price)
        r_mult = curr_dist / effective_risk
        
        # 1. Stall Check (Vol-Aware)
        if r_mult > min_profit_bank and p_dur > stall_threshold and abs(momentum) < (volatility * 0.3):
            results[i] = 1
            continue
            
        # 2. Reversal Check (Dynamic Momentum)
        if r_mult > 0.5: # In decent profit
            # If momentum flips against us by more than 0.5 ATR
            if p_type == 0 and momentum < -(volatility * 0.5): results[i] = 1
            elif p_type == 1 and momentum > (volatility * 0.5): results[i] = 1
            if results[i] == 1: continue
            
        # 3. Zombie Check (Time-Aware)
        if -0.2 < r_mult < 0 and p_dur > zombie_minutes:
            results[i] = 1
            
    return results

class TradeDoctor:
    """
    Active Trade Management (ATM) Agent.
    Monitors open positions for signs of weakness, stall, or reversal.
    Acts like a human trader managing trades post-entry.
    """

    def __init__(self, settings):
        self.settings = settings
        dyn = getattr(self.settings, "dynamic", {})
        doc_params = dyn.get("doctor_params", {})

        self.stall_threshold_minutes = doc_params.get("stall_threshold_minutes", 60.0)
        self.min_profit_to_bank = doc_params.get("min_profit_to_bank", 0.3)
        self.reversal_sensitivity = doc_params.get("reversal_sensitivity", 0.2)
        try:
            self.time_stop_bars = int(getattr(self.settings.risk, "time_stop_bars", 8) or 8)
        except Exception:
            self.time_stop_bars = 8
        tf = str(getattr(self.settings.system, "base_timeframe", "M5") or "M5").upper()
        tf_minutes = {
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
        }.get(tf, 5)
        self.time_stop_minutes = max(30.0, float(self.time_stop_bars) * float(tf_minutes))
        self._scaled_tickets: dict[int, bool] = {}

        self.exit_agent = ExitAgent(settings)
        try:
            self.exit_agent.load("models")
        except Exception as e:
            logger.warning(f"Trade analysis failed: {e}", exc_info=True)

        logger.info(
            f"TradeDoctor initialized with: Stall={self.stall_threshold_minutes:.1f}m, "
            f"BankR={self.min_profit_to_bank:.2f}, RevSens={self.reversal_sensitivity:.2f}, "
            f"TimeStop={self.time_stop_minutes:.1f}m"
        )

    def diagnose(self, positions: list[MT5Position], frames: dict[str, pd.DataFrame]) -> list[CloseInstruction]:
        """
        HPC Optimized: Evaluate all open positions using Numba.
        """
        instructions = []
        if not positions:
            return instructions

        symbol = self.settings.system.symbol
        target_positions = [p for p in positions if p.symbol == symbol]
        if not target_positions:
            return instructions

        df = frames.get("M1") or frames.get("M5")
        if df is None or df.empty:
            return instructions

        current_price = float(df["close"].iloc[-1])
        recent_closes = df["close"].iloc[-5:].values
        momentum = float(recent_closes[-1] - recent_closes[0]) if len(recent_closes) > 1 else 0.0
        rsi = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0
        vol = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.001
        now = datetime.now(UTC)

        # Prepare arrays for Numba
        pos_types = np.array([p.type for p in target_positions], dtype=np.int8)
        pos_prices_open = np.array([p.price_open for p in target_positions], dtype=np.float64)
        pos_sls = np.array([p.sl for p in target_positions], dtype=np.float64)
        pos_durations = np.array([(now - p.time).total_seconds() / 60.0 for p in target_positions], dtype=np.float64)

        # Call Numba Pass
        results = _diagnose_positions_numba(
            pos_types, pos_prices_open, pos_sls, pos_durations,
            current_price, momentum, rsi, vol,
            float(self.stall_threshold_minutes),
            float(self.min_profit_to_bank),
            float(self.reversal_sensitivity),
            float(self.time_stop_minutes),
        )

        # Convert results back to instructions
        close_all: set[int] = set()
        for i, res in enumerate(results):
            if res == 1:
                p = target_positions[i]
                close_all.add(int(p.ticket))
                instructions.append(CloseInstruction(
                    p.ticket, p.symbol, 0.0, "TradeDoctor: Technical Exit (Numba)", 1.0
                ))

        # Partial take-profit: scale out 50% once at >=1R if position not already marked for full close.
        for p in target_positions:
            ticket = int(p.ticket)
            if ticket in close_all:
                continue
            if self._scaled_tickets.get(ticket):
                continue
            try:
                risk_dist = abs(float(p.price_open) - float(p.sl)) if float(p.sl) > 0 else max(float(vol) * 2.0, 1e-9)
                if risk_dist <= 0:
                    continue
                current_dist = (
                    float(current_price) - float(p.price_open) if int(p.type) == 0 else float(p.price_open) - float(current_price)
                )
                r_multiple = current_dist / risk_dist
                if r_multiple < 1.0:
                    continue
                close_vol = round(max(0.01, float(p.volume) * 0.5), 2)
                if close_vol >= float(p.volume):
                    continue
                instructions.append(
                    CloseInstruction(
                        p.ticket,
                        p.symbol,
                        close_vol,
                        f"Scale-out 50% at {r_multiple:.2f}R",
                        float(r_multiple),
                    )
                )
                self._scaled_tickets[ticket] = True
            except Exception:
                continue

        live_tickets = {int(p.ticket) for p in target_positions}
        stale = [t for t in list(self._scaled_tickets.keys()) if t not in live_tickets]
        for t in stale:
            self._scaled_tickets.pop(t, None)

        return instructions

    def _check_position(
        self, pos: MT5Position, current_price: float, momentum: float, rsi: float, volatility: float, now: datetime
    ) -> CloseInstruction | None:
        """
        Diagnose a single position.
        """
        duration_mins = (now - pos.time).total_seconds() / 60.0

        risk_dist = abs(pos.price_open - pos.sl) if pos.sl > 0 else 0.0
        if risk_dist <= 0:
            return None  # Can't calc R without SL

        current_dist = (current_price - pos.price_open) if pos.type == 0 else (pos.price_open - current_price)
        r_multiple = current_dist / risk_dist

        state_vec = np.array(
            [
                r_multiple,
                duration_mins / 60.0,  # Normalize to hours roughly
                momentum / volatility if volatility > 0 else 0.0,
                volatility,
                rsi / 100.0,
                current_dist,
            ],
            dtype=np.float32,
        )

        rl_action = self.exit_agent.get_action(state_vec)

        self.exit_agent.observe_exit(pos.ticket, state_vec, rl_action, current_price, now)

        if rl_action > 0.5:  # FIX: Robust check for both binary (0/1) and continuous (0.0-1.0) actions
            return CloseInstruction(
                pos.ticket, pos.symbol, 0.0, f"AI Exit Agent Decision (R={r_multiple:.2f})", r_multiple
            )

        if r_multiple > self.min_profit_to_bank:  # At least evolved R profit
            # FIX: Use ATR (volatility) for stall detection instead of tight SL distance
            # If 5-bar momentum is less than 0.5 ATR, price is stalling.
            if duration_mins > self.stall_threshold_minutes and abs(momentum) < (volatility * 0.5):
                return CloseInstruction(
                    pos.ticket,
                    pos.symbol,
                    0.0,
                    f"Stalled in profit ({r_multiple:.2f}R) for {int(duration_mins)}m",
                    r_multiple,
                )

        if r_multiple > (self.min_profit_to_bank * 2.0):  # Good profit (2x min bank)
            thresh = risk_dist * self.reversal_sensitivity

            if pos.type == 0:  # Buy
                if momentum < -thresh:  # Sharp drop
                    return CloseInstruction(
                        pos.ticket, pos.symbol, 0.0, f"Momentum reversal in profit ({r_multiple:.2f}R)", r_multiple
                    )
            elif pos.type == 1:  # Sell
                if momentum > thresh:  # Sharp rally
                    return CloseInstruction(
                        pos.ticket, pos.symbol, 0.0, f"Momentum reversal in profit ({r_multiple:.2f}R)", r_multiple
                    )

        if -0.5 < r_multiple < 0:  # Small loss
            if duration_mins > self.time_stop_minutes:
                return CloseInstruction(
                    pos.ticket,
                    pos.symbol,
                    0.0,
                    f"Zombie trade (small loss) > {int(self.time_stop_minutes)}m",
                    r_multiple,
                )

        return None
