from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from forex_bot.strategy import evo_prop
from forex_bot.strategy.evo_prop import _convert_gpu_genome, _parse_gpu_devices


def test_parse_gpu_devices_normalizes_and_dedupes() -> None:
    out = _parse_gpu_devices("0, 2,2, -1, abc, 5")
    assert out == [0, 2, 5]


def test_convert_gpu_genome_maps_to_valid_strategy_gene() -> None:
    feature_names = ["RSI_14", "ADX_14", "EMA_20", "OPEN", "CLOSE"]
    available = {"RSI", "ADX", "EMA"}
    # tf_count=1, n_features=5, thresholds=2
    genome = [0.9, 0.8, -0.6, 0.5, 0.1, 0.05, 0.12, -0.08]
    gene = _convert_gpu_genome(
        genome=genome,
        fitness=12.3,
        feature_names=feature_names,
        available=available,
        max_indicators=3,
        threshold_scale=0.10,
        threshold_margin=0.02,
        threshold_clip=0.30,
        strategy_id="evogp_test_1",
    )
    assert gene is not None
    assert gene.strategy_id == "evogp_test_1"
    assert len(gene.indicators) >= 1
    assert len(gene.indicators) <= 3
    assert all(ind in available for ind in gene.indicators)
    assert gene.long_threshold > 0.0
    assert gene.short_threshold < 0.0
    assert gene.fitness == 12.3


def test_convert_gpu_genome_prefers_rust_score_ranking(monkeypatch) -> None:
    calls = {"rank": 0}

    def _rank_scores_desc(scores, absolute=False):
        calls["rank"] += 1
        assert bool(absolute) is True
        np.testing.assert_allclose(np.asarray(scores, dtype=np.float64), np.array([-0.6, 0.5, 0.1, 0.05, 0.12]))
        return np.array([0, 1, 4, 2, 3], dtype=np.int64)

    monkeypatch.setattr(evo_prop, "_fb", SimpleNamespace(rank_scores_desc=_rank_scores_desc), raising=False)

    feature_names = ["RSI_14", "ADX_14", "EMA_20", "OPEN", "CLOSE"]
    available = {"RSI", "ADX", "EMA"}
    genome = [0.9, -0.6, 0.5, 0.1, 0.05, 0.12, -0.08, 0.04]
    gene = _convert_gpu_genome(
        genome=genome,
        fitness=12.3,
        feature_names=feature_names,
        available=available,
        max_indicators=2,
        threshold_scale=0.10,
        threshold_margin=0.02,
        threshold_clip=0.30,
        strategy_id="evogp_test_rust_rank",
    )

    assert calls["rank"] == 1
    assert gene is not None
    assert gene.indicators[:2] == ["RSI", "ADX"]
