from __future__ import annotations

import sys

from forex_bot import main as main_mod


def test_main_train_mode_skips_missing_keys_warning(monkeypatch, capsys) -> None:
    async def _fake_main_async(_args=None):
        return None

    monkeypatch.setattr(main_mod, "main_async", _fake_main_async)
    monkeypatch.setattr(sys, "argv", ["forex_bot.main", "--train"])

    main_mod.main()

    captured = capsys.readouterr()
    assert "Keys file" not in captured.err
