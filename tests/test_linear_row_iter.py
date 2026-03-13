from __future__ import annotations

from tests._compat_pd import pd

from forex_bot.models import linear


def test_iter_feature_dict_rows_preserves_order_and_values():
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.5, -3.0],
            "f2": [10, 20, 30],
        }
    )
    rows = list(linear._iter_feature_dict_rows(df))
    assert rows == [
        {"f1": 1.0, "f2": 10.0},
        {"f1": 2.5, "f2": 20.0},
        {"f1": -3.0, "f2": 30.0},
    ]

