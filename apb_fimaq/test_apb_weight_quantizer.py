"""
Unit tests for APBWeightQuantizer:
  1. α, δ init from weight stats
  2. Mask all-False  →  forward returns w unchanged
  3. Mask all-True   →  forward returns ±α (only 2 unique values per group)
  4. Mixed mask      →  binary positions are ±α, FP positions are original w
  5. STE backward    →  ∂APB/∂w = 1 everywhere
  6. n_V=3 (QKV)     →  3 independent α values, per-group binarization
  7. set_mask_from_fim → low-FIM positions get masked
  8. get_hard_value  →  matches forward but no STE (for end-of-training bake)
"""
import sys
import torch

sys.path.insert(0, r'D:\xauduabo\Code+NCKH\Prune_QT_VITs')
from apb_fimaq import APBWeightQuantizer


def _passed(name): print(f'  [PASS] {name}')


def test_init_stats():
    w = torch.randn(1, 16, 8) * 0.05  # std ≈ 0.05
    q = APBWeightQuantizer(w, n_V=1)
    expected_alpha = w.abs().mean().item()
    expected_delta = 3.0 * w.std().item()
    assert abs(q.apb_alpha.item() - expected_alpha) < 1e-6
    assert abs(q.apb_delta.item() - expected_delta) < 1e-6
    _passed('init: α=mean|w|, δ=3·std(w)')


def test_mask_all_false():
    w = torch.randn(1, 16, 8)
    q = APBWeightQuantizer(w, n_V=1)
    out = q(w)
    assert torch.allclose(out, w), 'mask=False everywhere should return w unchanged'
    _passed('mask all-False: returns w unchanged')


def test_mask_all_true():
    w = torch.randn(1, 16, 8) * 0.1
    q = APBWeightQuantizer(w, n_V=1)
    q.set_mask(torch.ones_like(w, dtype=torch.bool))
    out = q(w)
    # STE trick `w + (w_apb-w).detach()` has FP noise, so use tolerance
    expected = q.apb_alpha.item() * torch.sign(w)
    assert torch.allclose(out, expected, atol=1e-5)
    assert torch.allclose(out.abs(), torch.full_like(out, q.apb_alpha.item()), atol=1e-5)
    _passed('mask all-True: only ±α values (FP tolerance)')


def test_mixed_mask():
    w = torch.randn(1, 8, 4)
    mask = torch.zeros_like(w, dtype=torch.bool)
    mask[0, :4] = True   # first half binary
    q = APBWeightQuantizer(w, n_V=1)
    q.set_mask(mask)
    out = q(w)
    # Binary positions (with FP tolerance from STE add-subtract)
    expected_bin = q.apb_alpha.item() * torch.sign(w[mask])
    assert torch.allclose(out[mask], expected_bin, atol=1e-5)
    # FP positions (should be exact since STE is w + (w_apb-w).detach() and w_apb==w here)
    assert torch.allclose(out[~mask], w[~mask], atol=1e-6)
    _passed('mixed mask: binary→±α, FP→w')


def test_ste_backward():
    w_init = torch.randn(1, 16, 8) * 0.1
    w = w_init.detach().clone().requires_grad_(True)  # fresh leaf with grad
    q = APBWeightQuantizer(w_init, n_V=1)
    q.set_mask(torch.rand_like(w_init) > 0.3)  # ~70% binary
    out = q(w)
    loss = out.sum()
    loss.backward()
    # ∂APB/∂w = 1 (STE) → grad should be all ones
    assert w.grad is not None, 'w.grad is None — w may not be a leaf tensor'
    assert torch.allclose(w.grad, torch.ones_like(w)), (
        f'STE failed: grad min={w.grad.min()}, max={w.grad.max()}')
    _passed('STE backward: ∂APB/∂w = 1 everywhere')


def test_n_V_3_qkv():
    # Simulate QKV weight: 3 groups (Q, K, V) with different scales
    w_q = torch.randn(16, 8) * 0.01
    w_k = torch.randn(16, 8) * 0.05
    w_v = torch.randn(16, 8) * 0.20
    w = torch.stack([w_q, w_k, w_v], dim=0)  # (3, 16, 8)
    q = APBWeightQuantizer(w, n_V=3)
    # Each group should get its own α
    assert q.apb_alpha.shape == (3,)
    assert q.apb_alpha[0] < q.apb_alpha[1] < q.apb_alpha[2], (
        f'α should increase with weight scale: {q.apb_alpha}')

    # Binarize all → each group uses its own α
    q.set_mask(torch.ones_like(w, dtype=torch.bool))
    out = q(w)
    for g in range(3):
        expected_g = q.apb_alpha[g].item() * torch.sign(w[g])
        assert torch.allclose(out[g], expected_g, atol=1e-5)
        assert torch.allclose(out[g].abs(), torch.full_like(out[g], q.apb_alpha[g].item()), atol=1e-5)
    _passed('n_V=3 QKV: per-group α, independent binarization')


def test_mask_from_fim():
    w = torch.randn(1, 16, 8)
    fim = torch.rand_like(w)  # random importance scores in [0,1]
    q = APBWeightQuantizer(w, n_V=1)
    q.set_mask_from_fim(fim, binary_ratio=0.8)
    # ~80% of positions should be masked (lowest 80% FIM)
    ratio = q.binary_ratio
    assert abs(ratio - 0.8) < 0.05, f'binary_ratio={ratio}, expected ~0.8'
    # The masked positions should have lower FIM than unmasked
    fim_binary = fim[q.mask]
    fim_fp = fim[~q.mask]
    assert fim_binary.max() <= fim_fp.min() + 1e-6
    _passed(f'set_mask_from_fim(0.8): got ratio={ratio:.3f}, threshold consistent')


def test_get_hard_value():
    w = torch.randn(1, 16, 8) * 0.1
    q = APBWeightQuantizer(w, n_V=1)
    q.set_mask(torch.rand_like(w) > 0.4)  # ~60% binary
    hard = q.get_hard_value(w)
    with torch.no_grad():
        soft = q(w)
    assert torch.allclose(hard, soft), 'hard and STE-forward should match in value'
    assert not hard.requires_grad
    _passed('get_hard_value: matches forward, no grad')


def test_repr_and_diagnostics():
    w = torch.randn(1, 16, 8) * 0.1
    q = APBWeightQuantizer(w, n_V=1)
    q.set_mask(torch.rand_like(w) > 0.3)
    eff_bits = q.storage_bits_per_weight()
    p = q.binary_ratio
    assert abs(eff_bits - (1.0 * p + 32.0 * (1 - p))) < 1e-6
    r = repr(q)
    assert 'APBWeightQuantizer' in r and 'α=' in r and 'binary_ratio' in r
    _passed(f'diagnostics: {r}')


if __name__ == '__main__':
    print('Running APBWeightQuantizer tests...\n')
    test_init_stats()
    test_mask_all_false()
    test_mask_all_true()
    test_mixed_mask()
    test_ste_backward()
    test_n_V_3_qkv()
    test_mask_from_fim()
    test_get_hard_value()
    test_repr_and_diagnostics()
    print('\nAll tests passed ✓')
