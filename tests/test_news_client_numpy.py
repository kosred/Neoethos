from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from forex_bot.data.news import client


def _analyzer_stub() -> client.SentimentAnalyzer:
    return client.SentimentAnalyzer.__new__(client.SentimentAnalyzer)


def test_build_features_strict_returns_frame_without_pandas(monkeypatch):
    analyzer = _analyzer_stub()
    events = [
        {
            "published_at": datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            "sentiment": 0.6,
            "confidence": 0.8,
        },
        {
            "published_at": datetime(2024, 1, 1, 0, 4, tzinfo=UTC),
            "sentiment": -0.2,
            "confidence": 0.7,
        },
    ]
    base_index = np.array(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T00:02:00",
            "2024-01-01T00:05:00",
        ],
        dtype="datetime64[ns]",
    )
    feat = analyzer.build_features(events, base_index, include_activation=True)
    assert hasattr(feat, "columns")
    assert hasattr(feat, "index")
    assert "news_sentiment" in feat.columns
    assert "news_confidence" in feat.columns
    assert "news_count" in feat.columns
    assert "news_recency_minutes" in feat.columns
    assert "news_nearby" in feat.columns
    np.testing.assert_allclose(np.asarray(feat["news_count"]), np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(feat["news_nearby"]), np.array([1, 1, 1], dtype=np.int8))


def test_build_features_prefers_rust_sorted_index_order(monkeypatch):
    calls = {"sort": 0}

    def _sorted_index_order(idx_ns):
        calls["sort"] += 1
        arr = np.asarray(idx_ns, dtype=np.int64)
        return np.argsort(arr, kind="mergesort")

    monkeypatch.setattr(client, "_fb", type("FB", (), {"sorted_index_order": staticmethod(_sorted_index_order)}))
    analyzer = _analyzer_stub()
    events = [
        {
            "published_at": datetime(2024, 1, 1, 0, 4, tzinfo=UTC),
            "sentiment": -0.2,
            "confidence": 0.7,
        },
        {
            "published_at": datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            "sentiment": 0.6,
            "confidence": 0.8,
        },
    ]
    base_index = np.array(
        [
            "2024-01-01T00:05:00",
            "2024-01-01T00:00:00",
            "2024-01-01T00:02:00",
        ],
        dtype="datetime64[ns]",
    )

    feat = analyzer.build_features(events, base_index, include_activation=False)

    assert calls["sort"] >= 2
    np.testing.assert_array_equal(
        np.asarray(feat.index),
        np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:02:00",
                "2024-01-01T00:05:00",
            ],
            dtype="datetime64[ns]",
        ),
    )
    np.testing.assert_allclose(np.asarray(feat["news_count"]), np.array([0.0, 1.0, 2.0], dtype=np.float32))


def test_build_features_prefers_rust_news_aggregate_helpers(monkeypatch):
    calls = {"sort": 0, "agg": 0, "act": 0}

    def _sorted_index_order(idx_ns):
        calls["sort"] += 1
        arr = np.asarray(idx_ns, dtype=np.int64)
        return np.argsort(arr, kind="mergesort")

    def _aggregate_news_features(base_idx_ns, event_idx_ns, event_sent, event_conf, lookback_ns):
        calls["agg"] += 1
        assert int(lookback_ns) == 6 * 60 * 1_000_000_000
        assert np.asarray(base_idx_ns, dtype=np.int64).shape[0] == 3
        assert np.asarray(event_idx_ns, dtype=np.int64).shape[0] == 2
        return (
            np.array([0.0, 0.6, -0.2], dtype=np.float32),
            np.array([0.0, 0.8, 0.7], dtype=np.float32),
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
            np.array([9999.0, 1.0, 1.0], dtype=np.float32),
        )

    def _aggregate_news_activation(base_idx_ns, event_idx_ns, event_sent, event_conf, back_ns, fwd_ns):
        calls["act"] += 1
        assert int(back_ns) == 60 * 60 * 1_000_000_000
        assert int(fwd_ns) == 15 * 60 * 1_000_000_000
        return (
            np.array([1, 1, 1], dtype=np.int8),
            np.array([0.8, 0.8, 0.8], dtype=np.float32),
            np.array([0.6, 0.6, 0.6], dtype=np.float32),
        )

    monkeypatch.setattr(
        client,
        "_fb",
        type(
            "FB",
            (),
            {
                "sorted_index_order": staticmethod(_sorted_index_order),
                "aggregate_news_features": staticmethod(_aggregate_news_features),
                "aggregate_news_activation": staticmethod(_aggregate_news_activation),
            },
        ),
    )

    analyzer = _analyzer_stub()
    events = [
        {
            "published_at": datetime(2024, 1, 1, 0, 4, tzinfo=UTC),
            "sentiment": -0.2,
            "confidence": 0.7,
        },
        {
            "published_at": datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            "sentiment": 0.6,
            "confidence": 0.8,
        },
    ]
    base_index = np.array(
        [
            "2024-01-01T00:05:00",
            "2024-01-01T00:00:00",
            "2024-01-01T00:02:00",
        ],
        dtype="datetime64[ns]",
    )

    feat = analyzer.build_features(events, base_index, include_activation=True)

    assert calls["sort"] >= 2
    assert calls["agg"] == 1
    assert calls["act"] == 1
    np.testing.assert_allclose(np.asarray(feat["news_sentiment"]), np.array([0.0, 0.6, -0.2], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(feat["news_count"]), np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(feat["news_nearby"]), np.array([1, 1, 1], dtype=np.int8))
