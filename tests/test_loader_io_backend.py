from __future__ import annotations

from forex_bot.data import loader


def test_frame_io_backend_prefers_new_env(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_FRAME_IO_BACKEND", "pyarrow")
    monkeypatch.setenv("FOREX_BOT_DATA_IO_BACKEND", "pandas")
    assert loader._frame_io_backend() == "pyarrow"


def test_frame_io_backend_supports_aliases(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_FRAME_IO_BACKEND", "pl")
    assert loader._frame_io_backend() == "polars"
    monkeypatch.setenv("FOREX_BOT_FRAME_IO_BACKEND", "python")
    assert loader._frame_io_backend() == "pandas"


def test_frame_io_backend_unknown_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_FRAME_IO_BACKEND", "not-a-backend")
    assert loader._frame_io_backend() == "auto"
