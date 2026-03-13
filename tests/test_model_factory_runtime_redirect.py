from __future__ import annotations

from types import SimpleNamespace

from forex_bot.training import model_factory as mf


class _DummyTransformerModel:
    def __init__(
        self,
        d_model: int = 0,
        n_heads: int = 0,
        n_layers: int = 0,
        idx: int = 0,
        device: str = "cpu",
    ) -> None:
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.idx = int(idx)
        self.device = str(device)


def _make_factory() -> mf.ModelFactory:
    fac = object.__new__(mf.ModelFactory)
    fac.settings = SimpleNamespace(
        models=SimpleNamespace(
            train_batch_size=0,
            transformer_d_model=320,
            transformer_n_heads=10,
            transformer_n_layers=6,
            nf_hidden_dim=777,
            tide_hidden_dim=256,
            nbeats_hidden_dim=256,
            kan_hidden_dim=256,
            tabnet_hidden_dim=64,
            max_epochs_by_model={},
        ),
        system=SimpleNamespace(enable_gpu=False, num_gpus=0, device="cpu"),
    )
    fac.models_dir = None
    fac.available_gpus = []
    fac.prefer_gpu = False
    return fac


def test_create_model_applies_runtime_redirect_before_class_lookup(monkeypatch):
    fac = _make_factory()
    called: dict[str, str] = {}

    def _fake_resolve(name: str) -> str:
        return "transformer" if name == "patchtst" else name

    def _fake_get_model_class(name: str, prefer_gpu: bool = False):  # noqa: ARG001
        called["name"] = name
        return _DummyTransformerModel

    monkeypatch.setattr(mf, "_resolve_runtime_model_name", _fake_resolve)
    monkeypatch.setattr(mf, "get_model_class", _fake_get_model_class)
    fac._configure_instance = lambda *args, **kwargs: None  # type: ignore[assignment]
    fac._maybe_warm_start = lambda *args, **kwargs: None  # type: ignore[assignment]

    model = fac.create_model("patchtst", best_params={}, idx=3)
    assert called["name"] == "transformer"
    assert isinstance(model, _DummyTransformerModel)
    assert model.d_model == 320
    assert model.n_heads == 10
    assert model.n_layers == 6
    assert model.idx == 3


def test_create_model_without_redirect_keeps_original_model_name(monkeypatch):
    fac = _make_factory()
    called: dict[str, str] = {}

    def _fake_get_model_class(name: str, prefer_gpu: bool = False):  # noqa: ARG001
        called["name"] = name
        return _DummyTransformerModel

    monkeypatch.setattr(mf, "_resolve_runtime_model_name", lambda name: name)
    monkeypatch.setattr(mf, "get_model_class", _fake_get_model_class)
    fac._configure_instance = lambda *args, **kwargs: None  # type: ignore[assignment]
    fac._maybe_warm_start = lambda *args, **kwargs: None  # type: ignore[assignment]

    model = fac.create_model("patchtst", best_params={}, idx=3)
    assert called["name"] == "patchtst"
    assert isinstance(model, _DummyTransformerModel)
    # Without redirect, patchtst presets set hidden_dim, not transformer d_model.
    assert model.d_model == 0

