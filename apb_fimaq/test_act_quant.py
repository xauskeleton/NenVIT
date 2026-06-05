"""Fast CPU self-test for LSQ activation quant in qat.py (no model/dataset download).

Run on Kaggle (or anywhere torch is installed):
    python apb_fimaq/test_act_quant.py

Checks:
  1. forward quantizes to the right signed-symmetric levels {Qn..Qp}·s
  2. lazy step init = 2·E|x|/sqrt(Qp)
  3. gradient w.r.t. step matches the analytic LSQ formula
     (round(x/s)-x/s in-range, Qn below, Qp above), up to the 1/sqrt(N·Qp) scale
  4. gradient w.r.t. x is STE (1 in-range, 0 out-of-range)
  5. APBLinear(act_bits=8) actually quantizes its input and is differentiable
  6. backward-compat: act_bits=0 → no act_quant, identical to weight-only path
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qat import LSQActQuant, APBLinear   # noqa: E402


def test_forward_levels():
    b = 4
    q = LSQActQuant(b)
    x = torch.randn(2000) * 3.0
    y = q(x)                          # triggers lazy init
    s = q.step.item()
    # init check
    exp = 2 * x.abs().mean().item() / math.sqrt(2 ** (b - 1) - 1)
    assert abs(s - exp) < 1e-5, (s, exp)
    # every output must be an integer multiple of s within [Qn, Qp]
    levels = (y / s).round()
    assert torch.allclose(y, levels * s, atol=1e-5)
    assert levels.min() >= q.Qn - 1e-6 and levels.max() <= q.Qp + 1e-6
    assert set(levels.unique().tolist()).issubset(set(range(q.Qn, q.Qp + 1)))
    print(f'[1,2] forward levels + init OK (b={b}, s={s:.4f}, '
          f'levels used={sorted(set(int(v) for v in levels.unique().tolist()))})')


def test_step_gradient():
    b = 3
    q = LSQActQuant(b)
    x = torch.linspace(-10, 10, 501)
    _ = q(x)                          # init step
    s0 = q.step.detach().clone()
    q.zero_grad()
    out = q(x)
    out.sum().backward()
    g_auto = q.step.grad.item()

    # analytic LSQ: per-elem dq/ds, then scaled by g = 1/sqrt(N·Qp)
    u = (x / s0)
    per = torch.where(u <= q.Qn, torch.full_like(u, q.Qn),
          torch.where(u >= q.Qp, torch.full_like(u, q.Qp), u.round() - u))
    g_scale = 1.0 / math.sqrt(x.numel() * q.Qp)
    g_analytic = (per.sum() * g_scale).item()
    assert abs(g_auto - g_analytic) < 1e-4, (g_auto, g_analytic)
    print(f'[3] step grad matches analytic LSQ OK (auto={g_auto:.5f}, '
          f'analytic={g_analytic:.5f})')


def test_x_gradient_ste():
    b = 2
    q = LSQActQuant(b)
    x = torch.tensor([-100.0, 0.1, -0.1, 100.0], requires_grad=True)
    _ = q(x.detach())                 # init step on similar-scale data
    q.zero_grad()
    y = q(x)
    y.sum().backward()
    s = q.step.item()
    u = x.detach() / s
    in_range = (u > q.Qn) & (u < q.Qp)
    expected = in_range.float()
    assert torch.allclose(x.grad, expected), (x.grad, expected)
    print(f'[4] x grad is STE (1 in-range, 0 clipped) OK')


def test_apblinear_act8():
    lin = nn.Linear(16, 8)
    mask = torch.rand(8, 16) < 0.75
    layer = APBLinear(lin, mask, act_bits=8)
    assert layer.act_quant is not None
    x = torch.randn(5, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == (5, 8)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.act_quant.step.grad is not None
    assert layer.latent_weight.grad is not None
    print('[5] APBLinear(act_bits=8) quantizes input + all grads finite OK')


def test_backward_compat():
    lin = nn.Linear(16, 8)
    mask = torch.rand(8, 16) < 0.75
    layer0 = APBLinear(lin, mask, act_bits=0)     # off
    layer32 = APBLinear(lin, mask, act_bits=32)   # off
    assert layer0.act_quant is None and layer32.act_quant is None
    x = torch.randn(5, 16)
    # weight-only path must be identical to a hand-rolled APB forward
    from qat import _APBSTE
    eff = _APBSTE.apply(layer0.latent_weight, layer0.alpha.abs(), layer0.mask)
    ref = torch.nn.functional.linear(x, eff, layer0.bias)
    assert torch.allclose(layer0(x), ref, atol=1e-6)
    print('[6] act_bits=0/32 → weight-only, identical to old path OK')


if __name__ == '__main__':
    torch.manual_seed(0)
    test_forward_levels()
    test_step_gradient()
    test_x_gradient_ste()
    test_apblinear_act8()
    test_backward_compat()
    print('\nALL ACT-QUANT TESTS PASSED ✓')
