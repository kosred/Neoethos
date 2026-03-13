import numpy as np
from tests._compat_pd import pd

from forex_bot.strategy import evo_prop as ep


class _Models:
    prop_search_train_years = 0


class _Settings:
    models = _Models()


def test_trim_to_recent_years_keeps_recent_slice_and_attrs():
    idx = pd.date_range("2010-01-01", periods=15, freq="365D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.linspace(1.0, 2.0, len(idx)),
            "high": np.linspace(1.1, 2.1, len(idx)),
            "low": np.linspace(0.9, 1.9, len(idx)),
            "close": np.linspace(1.0, 2.0, len(idx)),
        },
        index=idx,
    )
    df.attrs["symbol"] = "EURUSD"

    out = ep._trim_to_recent_years(df, 10.0)
    assert len(out) < len(df)
    assert str(out.attrs.get("symbol", "")) == "EURUSD"
    assert out.index.min() >= (df.index.max() - pd.Timedelta(days=365.2425 * 10.0))


def test_train_years_cfg_prefers_env(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_PROP_SEARCH_TRAIN_YEARS", "10")
    settings = _Settings()
    settings.models.prop_search_train_years = 7
    years = ep._train_years_cfg(settings)
    assert abs(years - 10.0) < 1e-9

