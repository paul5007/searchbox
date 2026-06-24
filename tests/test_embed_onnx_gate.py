"""Regression test for #18 (R06a): EMBED_ONNX gated embedding backend, fail-closed.

The default embedder (jina-embeddings-v5-text-small) hard-rejects backend='onnx' upstream
(custom_st.py raises ValueError) and ships no onnx weights. When EMBED_ONNX=1 the loader MUST
surface that as a structured error telling the operator to set EMBED_ONNX=0 — it must NOT silently
serve fp32 the operator did not ask for (that hides the misconfig and masks the missing speedup).

Invariants:
  1. EMBED_ONNX=1 + a model that can't load via onnx  -> RuntimeError naming EMBED_ONNX, not the
     raw library error, and not a silent torch fallback.
  2. EMBED_ONNX=0 (default) -> the plain torch loader path (no backend/provider kwargs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))


def _fresh_module(monkeypatch_env: dict):
    """Import dataroom_service with a controlled env (module-level flags read at import)."""
    import importlib, os
    for k, v in monkeypatch_env.items():
        os.environ[k] = v
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # keep the loader on CPU, deterministic
    import dataroom_service as d
    importlib.reload(d)
    return d


def _fake_st_module(behavior):
    """A stand-in `sentence_transformers` module whose SentenceTransformer runs `behavior`."""
    import types
    mod = types.ModuleType("sentence_transformers")

    class _ST:
        def __init__(self, *a, **kw):
            self.args, self.kwargs = a, kw
            behavior(a, kw, self)

    mod.SentenceTransformer = _ST
    return mod


def test_onnx_on_failclosed_raises_structured_error():
    d = _fresh_module({"EMBED_ONNX": "1"})
    assert d.EMBED_ONNX is True
    d._embed_model = None
    d._want_cuda = lambda: False

    def boom(a, kw, self):
        # mimic the upstream ValueError jina-v5 raises for backend='onnx'
        raise ValueError("Backend 'onnx' is not supported, please use 'torch' instead")

    sys.modules["sentence_transformers"] = _fake_st_module(boom)
    try:
        d.embed_model()
        assert False, "expected RuntimeError, got none (fail-OPEN)"
    except RuntimeError as e:
        msg = str(e)
        assert "EMBED_ONNX=1" in msg, msg
        assert "Set EMBED_ONNX=0" in msg, msg
        # the raw upstream cause is preserved, not swallowed
        assert "onnx' is not supported" in msg
    finally:
        sys.modules.pop("sentence_transformers", None)


def test_onnx_on_passes_backend_kwargs():
    d = _fresh_module({"EMBED_ONNX": "1"})
    d._embed_model = None
    d._want_cuda = lambda: False
    seen = {}

    def capture(a, kw, self):
        seen["args"], seen["kwargs"] = a, kw

    sys.modules["sentence_transformers"] = _fake_st_module(capture)
    try:
        d.embed_model()
        assert seen["kwargs"].get("backend") == "onnx", seen["kwargs"]
        assert seen["kwargs"]["model_kwargs"]["provider"] == "CPUExecutionProvider"
    finally:
        sys.modules.pop("sentence_transformers", None)
        d._embed_model = None


def test_onnx_off_uses_plain_torch_loader():
    d = _fresh_module({"EMBED_ONNX": "0"})
    assert d.EMBED_ONNX is False
    d._embed_model = None
    d._want_cuda = lambda: False
    seen = {}

    def capture(a, kw, self):
        seen["kwargs"] = kw

    sys.modules["sentence_transformers"] = _fake_st_module(capture)
    try:
        d.embed_model()
        # torch path: no onnx backend, no provider
        assert "backend" not in seen["kwargs"], seen["kwargs"]
        assert seen["kwargs"].get("device") == "cpu"
    finally:
        sys.modules.pop("sentence_transformers", None)
        d._embed_model = None


def test_quant_int8_selects_quant_graph():
    d = _fresh_module({"EMBED_ONNX": "1", "EMBED_QUANT": "int8"})
    assert d.EMBED_QUANT == "int8"
    d._embed_model = None
    d._want_cuda = lambda: False
    seen = {}
    sys.modules["sentence_transformers"] = _fake_st_module(
        lambda a, kw, self: seen.update(kw))
    try:
        d.embed_model()
        assert seen["model_kwargs"]["file_name"] == "model_qint8_avx512_vnni.onnx", seen
    finally:
        sys.modules.pop("sentence_transformers", None)
        d._embed_model = None


def test_quant_unknown_level_failcloses():
    d = _fresh_module({"EMBED_ONNX": "1", "EMBED_QUANT": "int4"})
    d._embed_model = None
    d._want_cuda = lambda: False
    sys.modules["sentence_transformers"] = _fake_st_module(lambda a, kw, self: None)
    try:
        d.embed_model()
        assert False, "expected RuntimeError for unknown EMBED_QUANT"
    except RuntimeError as e:
        assert "unknown" in str(e) and "int4" in str(e), str(e)
    finally:
        sys.modules.pop("sentence_transformers", None)
        d._embed_model = None


def test_no_quant_omits_file_name():
    d = _fresh_module({"EMBED_ONNX": "1", "EMBED_QUANT": ""})
    assert d.EMBED_QUANT == ""
    d._embed_model = None
    d._want_cuda = lambda: False
    seen = {}
    sys.modules["sentence_transformers"] = _fake_st_module(
        lambda a, kw, self: seen.update(kw))
    try:
        d.embed_model()
        assert "file_name" not in seen["model_kwargs"], seen
    finally:
        sys.modules.pop("sentence_transformers", None)
        d._embed_model = None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all EMBED_ONNX gate tests passed")
