from __future__ import annotations

import copy
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from ..models.base import ExpertModel
from ..models.trees import LightGBMExpert

logger = logging.getLogger(__name__)


def _is_dataframe_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "index") and hasattr(value, "iloc"))


def _is_frame_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "index") and hasattr(value, "__getitem__"))


def _frame_columns(value: Any) -> list[str]:
    cols = getattr(value, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _is_series_like(value: Any) -> bool:
    return bool(hasattr(value, "index") and hasattr(value, "to_numpy") and not hasattr(value, "columns"))


class OnlineLearner:
    """
    Handles incremental learning (refitting) of models based on real-time trade outcomes.
    Also implements 'Novelty/History Guard' to prevent repeating past mistakes.
    """

    def __init__(
        self,
        models_dir: str,
        buffer_size: int = 1000,
        min_samples_for_update: int = 50,
        update_frequency: int = 10,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.buffer_size = buffer_size
        self.min_samples_for_update = min_samples_for_update
        self.update_frequency = update_frequency  # Update every N samples added

        self.buffer: deque = deque(maxlen=buffer_size)
        self.samples_since_update = 0

        self.loss_archive: list[np.ndarray] = []
        self.loss_archive_limit = 5000

        self.models: dict[str, ExpertModel] = {}
        self.scaler = None

    def load_models(self, models_dict: dict[str, ExpertModel]) -> None:
        self.models = models_dict
        bundle_path = self.models_dir / "models.joblib"
        try:
            bundle = joblib.load(bundle_path)
            if isinstance(bundle, dict):
                self.scaler = bundle.get("scaler")
            else:
                self.scaler = None
        except FileNotFoundError:
            self.scaler = None
        except EOFError as exc:
            # Corrupt/partial file (common when interrupted while writing). Repair best-effort and continue.
            self.scaler = None
            logger.info(f"models.joblib is unreadable ({exc}); repairing thin bundle and continuing.")
            self._repair_models_bundle()
        except Exception as exc:
            self.scaler = None
            logger.info(f"Online learner could not load models bundle ({bundle_path}): {exc}")

    def _repair_models_bundle(self) -> None:
        """
        Best-effort repair of `models.joblib` to avoid repeated EOFErrors at startup.
        """
        bundle_path = self.models_dir / "models.joblib"
        payload = {"models": {}, "model_names": list(self.models.keys()), "scaler": None, "schema_version": 2}
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            tmp = bundle_path.with_name(bundle_path.name + ".tmp")
            joblib.dump(payload, tmp)
            tmp.replace(bundle_path)
        except Exception:
            return

    @staticmethod
    def _coerce_numeric_array(values: Any) -> np.ndarray:
        arr = np.asarray(values)
        if arr.size <= 0:
            return arr.astype(np.float32, copy=False)
        if arr.dtype.kind in {"b", "i", "u", "f"}:
            return arr.astype(np.float32, copy=False)
        flat = arr.reshape(-1)
        out = np.empty(flat.shape[0], dtype=np.float32)
        for i, val in enumerate(flat):
            try:
                out[i] = float(val)
            except Exception:
                out[i] = np.nan
        return out.reshape(arr.shape)

    @staticmethod
    def _as_feature_matrix(x: Any) -> np.ndarray:
        if x is None:
            return np.zeros((0, 0), dtype=np.float32)
        if _is_dataframe_like(x):
            try:
                arr = x.to_numpy(dtype=np.float32, copy=False)
            except Exception:
                try:
                    arr = OnlineLearner._coerce_numeric_array(x.to_numpy(copy=False))
                except Exception:
                    arr = OnlineLearner._coerce_numeric_array(np.asarray(x))
        elif _is_frame_like(x):
            cols = _frame_columns(x)
            mats: list[np.ndarray] = []
            n_rows = 0
            for col in cols:
                try:
                    vec = OnlineLearner._coerce_numeric_array(x[col]).reshape(-1)
                    mats.append(vec.astype(np.float32, copy=False))
                    n_rows = max(n_rows, int(vec.size))
                except Exception:
                    continue
            if mats and n_rows > 0:
                arr = np.zeros((n_rows, len(mats)), dtype=np.float32)
                for j, vec in enumerate(mats):
                    take = min(n_rows, int(vec.size))
                    if take > 0:
                        arr[:take, j] = vec[:take]
            else:
                arr = np.asarray(x)
        else:
            arr = np.asarray(x)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        out = np.asarray(arr, dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _as_labels(y: Any) -> np.ndarray:
        if y is None:
            return np.zeros((0,), dtype=np.int8)
        if _is_series_like(y):
            arr = np.asarray(y.to_numpy(copy=False))
        elif _is_frame_like(y):
            cols = _frame_columns(y)
            if cols:
                arr = np.asarray(y[cols[0]])
            else:
                arr = np.asarray(y)
        else:
            arr = np.asarray(y)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        else:
            arr = arr.reshape(-1)
        labels = np.asarray(arr, dtype=np.float32)
        labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
        return labels.astype(np.int8, copy=False)

    def add_sample(self, x: Any, y: Any, weight: float = 1.0) -> None:
        """
        Add a completed trade result to the learning buffer.
        y: 1 (Win), -1 (Loss), 0 (Neutral)
        """
        x_arr = self._as_feature_matrix(x)
        y_arr = self._as_labels(y)
        if x_arr.shape[0] == 0 or y_arr.size == 0:
            return

        if int(y_arr[0]) < 0:  # Loss
            try:
                vec = np.asarray(x_arr[0], dtype=np.float32)
                self._add_to_loss_archive(vec)
            except Exception as e:
                logger.warning(f"Failed to archive loss vector: {e}")

        rows = min(int(x_arr.shape[0]), int(y_arr.size))
        self.buffer.append((x_arr[:rows], y_arr[:rows], weight))
        self.samples_since_update += 1

        if self.samples_since_update >= self.update_frequency and len(self.buffer) >= self.min_samples_for_update:
            self.update_models()
            self.samples_since_update = 0

    def _add_to_loss_archive(self, vector: np.ndarray) -> None:
        """Store bad decision vectors to avoid repeating them."""
        if len(self.loss_archive) >= self.loss_archive_limit:
            self.loss_archive.pop(0)
        self.loss_archive.append(np.asarray(vector, dtype=np.float32).reshape(-1))

    @staticmethod
    def _max_update_batch() -> int:
        try:
            val = int(os.environ.get("FOREX_BOT_ONLINE_MAX_BATCH", "512") or 512)
        except Exception:
            val = 512
        return max(32, val)

    def is_repeat_mistake(self, current_features: Any, threshold: float = 0.95) -> bool:
        """
        HPC Optimized: GPU-accelerated similarity guard.
        """
        if not self.loss_archive:
            return False

        try:
            current = self._as_feature_matrix(current_features)
            if current.shape[0] == 0:
                return False
            curr_np = np.asarray(current[0], dtype=np.float32).reshape(-1)
            if curr_np.size == 0:
                return False
            aligned_archive = [v for v in self.loss_archive if int(np.asarray(v).size) == int(curr_np.size)]
            if not aligned_archive:
                return False
            device = "cuda" if torch.cuda.is_available() else "cpu"
            curr_vec = torch.from_numpy(curr_np).to(device)
            archive_tensor = torch.from_numpy(np.stack(aligned_archive)).to(device)

            # HPC: Compute cosine similarity using pure torch matrix math
            # CosSim(A, B) = (A dot B) / (||A|| * ||B||)
            a_norm = F.normalize(curr_vec.unsqueeze(0), p=2, dim=1)
            b_norm = F.normalize(archive_tensor, p=2, dim=1)
            
            sims = torch.mm(a_norm, b_norm.transpose(0, 1))
            max_sim = torch.max(sims).item()

            if max_sim > threshold:
                logger.warning(f"Similarity guard: current setup matches a past loss ({max_sim:.2f}). Blocking.")
                return True
        except Exception as e:
            logger.warning(f"GPU similarity check failed: {e}")

        return False

    def update_models(self) -> bool:
        """
        Perform incremental training (refitting) on all supported models.
        """
        if len(self.buffer) < self.min_samples_for_update:
            return False

        logger.info("Online Learning: updating models with %s recent trades...", len(self.buffer))

        x_list, y_list, _w_list = zip(*self.buffer, strict=False)
        max_batch = self._max_update_batch()
        if len(x_list) > max_batch:
            x_list = x_list[-max_batch:]
            y_list = y_list[-max_batch:]
        try:
            x_blocks: list[np.ndarray] = []
            y_blocks: list[np.ndarray] = []
            for x_item, y_item in zip(x_list, y_list, strict=False):
                x_arr = self._as_feature_matrix(x_item)
                y_arr = self._as_labels(y_item)
                if x_arr.shape[0] == 0 or y_arr.size == 0:
                    continue
                rows = min(int(x_arr.shape[0]), int(y_arr.size))
                if rows <= 0:
                    continue
                x_blocks.append(x_arr[:rows])
                y_blocks.append(y_arr[:rows])
            if not x_blocks or not y_blocks:
                return False
            x_batch = np.concatenate(x_blocks, axis=0).astype(np.float32, copy=False)
            y_batch = np.concatenate(y_blocks, axis=0).astype(np.int8, copy=False)
        except Exception as e:
            logger.error(f"Online batch preparation failed: {e}")
            return False

        if int(np.unique(y_batch).size) < 2:
            logger.info("Online learning skipped: update batch has <2 classes.")
            return False

        updated_count = 0

        for name, model in self.models.items():
            backup_state = None
            try:
                if isinstance(model, LightGBMExpert):
                    if model.model is not None and hasattr(model.model, "booster_"):
                        backup_state = copy.deepcopy(model.model)
                        model.model.fit(
                            x_batch,
                            y_batch,
                            init_model=model.model.booster_,
                            verbose=False,
                        )
                        updated_count += 1

                elif hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
                    backup_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
                    self._update_pytorch_model(model, x_batch, y_batch)
                    updated_count += 1

            except Exception as e:
                if backup_state is not None:
                    with np.errstate(all="ignore"):
                        try:
                            if isinstance(model, LightGBMExpert):
                                model.model = backup_state
                            elif hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
                                model.model.load_state_dict(backup_state, strict=True)
                        except Exception:
                            pass
                logger.warning(f"Failed to update {name}: {e}")

        logger.info(f"Successfully updated {updated_count} models.")
        return True

    def _update_pytorch_model(self, expert: ExpertModel, x: np.ndarray, y: np.ndarray) -> None:
        """Standard SGD step for PyTorch models."""
        model = expert.model
        device = getattr(expert, "device", "cpu")

        model.train()
        x_tens = torch.as_tensor(x, dtype=torch.float32, device=device)

        y_mapped = np.asarray(y, dtype=np.int64) + 1
        y_tens = torch.as_tensor(y_mapped, dtype=torch.long, device=device)

        criterion = torch.nn.CrossEntropyLoss()

        # Persist optimizer state to maintain momentum
        if not hasattr(expert, "_online_optimizer"):
            expert._online_optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

        optimizer = expert._online_optimizer

        optimizer.zero_grad(set_to_none=True)
        try:
            use_amp = torch.cuda.is_available() and device != "cpu" and hasattr(torch, "amp")
            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    out = model(x_tens)
                    loss = criterion(out, y_tens)
            else:
                out = model(x_tens)
                loss = criterion(out, y_tens)

            loss.backward()
            optimizer.step()
        except Exception as e:
            logger.warning(f"PyTorch update step failed: {e}")

    def snapshot(self) -> dict[str, Any]:
        """Return serializable state."""
        snap_buffer = []
        for i in range(min(len(self.buffer), 100)):
            snap_buffer.append(self.buffer[i])

        return {
            "loss_archive": [v.tolist() for v in self.loss_archive],
            "buffer_sample_count": len(self.buffer),
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Restore state from snapshot."""
        try:
            if "loss_archive" in state:
                self.loss_archive = [np.array(v) for v in state["loss_archive"]]
        except Exception as e:
            logger.warning(f"Online learning update failed: {e}", exc_info=True)

