import numpy as np
import pytest

from bittrellis.quant.formats import dequantize_nvfp4, quantize_nvfp4
from bittrellis.quantizers import nvfp4_blockfit as bf
from bittrellis.quantizers import nvfp4_gptq as gq
from bittrellis.quantizers import nvfp4_gptq_dn as dn
from bittrellis.synthetic import make_tiny_calibration

QUIET = {"log": lambda *_: None, "verify": False}


def _weights(seed, shape):
    return (np.random.default_rng(seed).standard_t(5, size=shape) * 0.02).astype(np.float32)


def _down_hessian(seed, n, tokens=2048):
    """mean x xᵀ of a SwiGLU-like input: correlated, heavy-tailed, a few channels far larger than the rest."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((tokens, n)) @ (rng.standard_normal((n, n)) * 0.15 + np.eye(n))
    x = g / (1 + np.exp(-g)) * rng.standard_normal((tokens, n))          # silu(gate) * up
    x *= np.exp(0.8 * rng.standard_normal(n))
    return (x.T @ x / tokens).astype(np.float32)


def _out_err(w, result, h):
    e = w.astype(np.float64) - dequantize_nvfp4(*result)
    return float(np.einsum("ij,jk,ik->", e, h.astype(np.float64), e))


def _same(a, b):
    return all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(a, b, strict=True))


def _same_tensors(a, b):
    """Two encoder outputs [(suffix, dtype, shape, data), ...] are the same bytes."""
    return [t[:3] for t in a] == [t[:3] for t in b] and _same([t[3] for t in a], [t[3] for t in b])


def test_bytes_repeat():
    w, h = _weights(1, (32, 512)), _down_hessian(1, 512)
    assert _same(dn.quantize(w, h), dn.quantize(w, h))


def test_factor_and_bytes_match_nvfp4_gptq():
    # the lean factor frees memory only: same arithmetic, same bytes as nvfp4_gptq's
    w, h = _weights(2, (24, 256)), _down_hessian(2, 256)
    assert dn.inverse_hessian_factor(h).tobytes() == gq.inverse_hessian_factor(h).tobytes()
    assert _same(dn.quantize(w, h), gq.quantize(w, h))


def test_down_shaped_gptq_beats_rtn_and_blockfit():
    # down_proj is wide (cols = intermediate > rows = hidden); output error under H must fall well below RTN
    w, h = _weights(3, (32, 512)), _down_hessian(3, 512)
    ours, fit, rtn = dn.quantize(w, h), bf.quantize(w, "mlp"), quantize_nvfp4(w)
    assert ours[2] == rtn[2] == fit[2]
    e_ours, e_fit, e_rtn = (_out_err(w, r, h) for r in (ours, fit, rtn))
    assert e_ours < 0.8 * e_rtn
    assert e_ours < e_fit


def test_rejects_a_mismatched_hessian():
    with pytest.raises(ValueError):
        dn.quantize(_weights(4, (4, 64)), np.eye(32, dtype=np.float32))


# ------------------------------------------------------------------ the encoder on the tiny model

def _ctx(base, calib):
    from bittrellis.quantizers import QuantContext
    from bittrellis.safetensors_io import SafeTensorsDir

    return QuantContext(base=SafeTensorsDir(base), sources={}, calibration=SafeTensorsDir(calib))


def _mlp_units():
    from bittrellis.model.qwen38 import Qwen38Arch
    from bittrellis.synthetic import TINY_TEXT

    return [u for u in Qwen38Arch.from_config({"text_config": TINY_TEXT}).units() if u.kind == "mlp"]


def _encode_all(q, ctx):
    return {lin.prefix: q.encode(ctx, u, lin, "NVFP4") for u in _mlp_units() for lin in u.linears}


@pytest.fixture
def served_down(monkeypatch):
    """Stand-in for an epoch that pins down_proj statistics: the test's own seeded Hessian per down_proj,
    served through the encoder's only statistics entry point. Production code never builds one."""
    real = dn.input_hessian
    stats: dict[str, np.ndarray] = {}

    def fake(ctx, lin):
        if lin.prefix.endswith(".down_proj"):
            if lin.prefix not in stats:                   # seeded by layer: build workers fork and agree
                layer = int(lin.prefix.split(".layers.")[1].split(".")[0])
                stats[lin.prefix] = _down_hessian(10 + layer, lin.cols, tokens=512)
            return stats[lin.prefix]
        return real(ctx, lin)

    monkeypatch.setattr(dn, "input_hessian", fake)
    return stats


def _no_down(real):
    return lambda ctx, lin: None if lin.prefix.endswith(".down_proj") else real(ctx, lin)


def _gate_up_and_fit(ctx):
    """Reference bytes: nvfp4_gptq for gate/up (it has no down_proj statistics path), blockfit for down."""
    gptq, fit = gq.NVFP4GPTQ(), bf.NVFP4BlockFit()
    return {lin.prefix: (fit if lin.prefix.endswith(".down_proj") else gptq).encode(ctx, u, lin, "NVFP4")
            for u in _mlp_units() for lin in u.linears}


def test_without_down_statistics_it_is_nvfp4_gptq(tiny_all, tmp_path, monkeypatch):
    # epoch hpc01-e6: down_proj falls back to blockfit, gate/up are nvfp4_gptq's bytes exactly
    monkeypatch.setattr(dn, "input_hessian", _no_down(dn.input_hessian))
    monkeypatch.setattr(gq, "input_hessian", _no_down(gq.input_hessian))
    calib = make_tiny_calibration(tmp_path / "calib", seed=0)
    ours = _encode_all(dn.NVFP4GPTQDown(), _ctx(tiny_all[0], calib))
    theirs = _encode_all(gq.NVFP4GPTQ(), _ctx(tiny_all[0], calib))
    assert ours.keys() == theirs.keys() and all(_same_tensors(ours[k], theirs[k]) for k in ours)


def test_down_statistics_are_used_when_served(tiny_all, tmp_path, monkeypatch, served_down):
    from bittrellis.quantizers.base import f32_weight

    calib = make_tiny_calibration(tmp_path / "calib", seed=0)
    first = _encode_all(dn.NVFP4GPTQDown(), _ctx(tiny_all[0], calib))
    again = _encode_all(dn.NVFP4GPTQDown(), _ctx(tiny_all[0], calib))
    assert all(_same_tensors(first[k], again[k]) for k in first)

    monkeypatch.setattr(gq, "input_hessian", _no_down(gq.input_hessian))
    ctx = _ctx(tiny_all[0], calib)
    ref = _gate_up_and_fit(ctx)
    for k in first:
        if k.endswith(".down_proj"):
            assert not _same_tensors(first[k], ref[k])               # down: GPTQ, not blockfit
        else:
            assert _same_tensors(first[k], ref[k])                   # gate/up: nvfp4_gptq's bytes
    for lin in (lin for u in _mlp_units() for lin in u.linears if lin.prefix.endswith(".down_proj")):
        ours = first[lin.prefix]
        w, h = f32_weight(ctx, lin), served_down[lin.prefix]
        result = (ours[0][3], ours[1][3], np.float32(ours[2][3]))
        assert _out_err(w, result, h) < _out_err(w, bf.quantize(w, "mlp"), h)
        assert _out_err(w, result, h) < _out_err(w, quantize_nvfp4(w), h)


def test_tiny_build_is_repeatable_and_audited(tiny_all, tmp_path, monkeypatch, served_down):
    from bittrellis.build import build
    from bittrellis.manifest import Manifest
    from bittrellis.track import load_track
    from bittrellis.validate import audit

    monkeypatch.setenv("BITTRELLIS_REPLAY_CACHE", str(tmp_path / "cache"))
    manifest = Manifest.from_dict({
        "schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "test-gptq-dn", "default": "NVFP4",
        "rules": [{"match": "L*.mlp", "format": "NVFP4", "quantizer": "nvfp4_gptq_dn"},
                  {"match": "L*.attn.*", "format": "NVFP4", "quantizer": "nvfp4_blockfit"},
                  {"match": "L*.gdn.*", "format": "NVFP4", "quantizer": "nvfp4_blockfit"}],
    })
    calib = make_tiny_calibration(tmp_path / "calib", seed=0)
    sources = {**dict(zip(("base", "gittensor_nvfp4", "unsloth_nvfp4"), tiny_all, strict=True)), "calibration": calib}
    record = build(manifest, load_track("HPC-01"), sources, tmp_path / "first", **QUIET)
    again = build(manifest, load_track("HPC-01"), sources, tmp_path / "second", **QUIET)
    assert record["files"] == again["files"]
    result = audit(tmp_path / "first", manifest, sources, verify_sources=False)
    assert result.ok, result.errors
    assert result.lineage["nvfp4_gptq_dn@v1"]["replayed_tensors"] > 0


def test_refuses_without_the_statistics(tiny_all):
    from bittrellis.quantizers import QuantContext
    from bittrellis.safetensors_io import SafeTensorsDir

    u = _mlp_units()[0]
    with pytest.raises(ValueError, match="needs the calibration"):
        dn.NVFP4GPTQDown().encode(QuantContext(base=SafeTensorsDir(tiny_all[0]), sources={}), u, u.linears[0], "NVFP4")
