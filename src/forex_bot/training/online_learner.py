from __future__ import annotations

import copy
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from ..models.base import ExpertModel
from ..models.trees import LightGBMExpert

logger = logging.getLogger(__name__)


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

    def add_sample(self, x: pd.DataFrame, y: pd.Series, weight: float = 1.0) -> None:
        """
        Add a completed trade result to the learning buffer.
        y: 1 (Win), -1 (Loss), 0 (Neutral)
        """
        if y.iloc[0] < 0:  # Loss
            try:
                vec = x.iloc[0].to_numpy(dtype=np.float32)
                self._add_to_loss_archive(vec)
            except Exception as e:
                logger.warning(f"Failed to archive loss vector: {e}")

        self.buffer.append((x, y, weight))
        self.samples_since_update += 1

        if self.samples_since_update >= self.update_frequency and len(self.buffer) >= self.min_samples_for_update:
            self.update_models()
            self.samples_since_update = 0

    def _add_to_loss_archive(self, vector: np.ndarray) -> None:
        """Store bad decision vectors to avoid repeating them."""
        if len(self.loss_archive) >= self.loss_archive_limit:
            self.loss_archive.pop(0)
        self.loss_archive.append(vector)

    @staticmethod
    def _max_update_batch() -> int:
        try:
            val = int(os.environ.get("FOREX_BOT_ONLINE_MAX_BATCH", "512") or 512)
        except Exception:
            val = 512
        return max(32, val)

    def is_repeat_mistake(self, current_features: pd.DataFrame, threshold: float = 0.95) -> bool:
        """
        HPC Optimized: GPU-accelerated similarity guard.
        """
        if not self.loss_archive:
            return False

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            curr_vec = torch.from_numpy(current_features.iloc[0].to_numpy(dtype=np.float32)).to(device)
            archive_tensor = torch.from_numpy(np.stack(self.loss_archive)).to(device)

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
            x_batch = pd.concat(x_list, axis=0, copy=False)
            y_batch = pd.concat(y_list, axis=0)
            non_numeric = [c for c in x_batch.columns if not pd.api.types.is_numeric_dtype(x_batch[c])]
            for col in non_numeric:
                x_batch[col] = pd.to_numeric(x_batch[col], errors="coerce")
            x_batch = x_batch.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            with np.errstate(all="ignore"):
                x_batch = x_batch.astype(np.float32, copy=False)
            y_batch = pd.to_numeric(y_batch, errors="coerce").fillna(0).astype(np.int8, copy=False)
        except Exception as e:
            logger.error(f"Online batch preparation failed: {e}")
            return False

        if int(pd.Series(y_batch).nunique(dropna=False)) < 2:
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

    def _update_pytorch_model(self, expert: ExpertModel, x: pd.DataFrame, y: pd.Series) -> None:
        """Standard SGD step for PyTorch models."""
        model = expert.model
        device = getattr(expert, "device", "cpu")

        model.train()
        x_tens = torch.as_tensor(x.values, dtype=torch.float32, device=device)

        y_mapped = y.values + 1
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
