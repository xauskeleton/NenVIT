"""
Minimal debug for FIMWeightExtractor only.
No FIMA-Q wrap — test on RAW Swin-S to verify backward hooks fire.
If FIM is non-zero here but zero with FIMA-Q wrap → wrap blocks gradient.
"""
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / 'apb_fimaq', PROJECT_ROOT / 'scripts'):
    sys.path.insert(0, str(p))

import timm  # noqa: E402
from tiny_imagenet_loader import TinyImageNetLoaderGenerator  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Loading raw Swin-S (no quant wrap)...')
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    model.to(device).eval()

    # Pick just 3 target Linears (avoid 96-layer flood in debug)
    pick = [
        'layers.0.blocks.0.mlp.fc1',
        'layers.2.blocks.0.mlp.fc1',
        'layers.3.blocks.1.mlp.fc2',
    ]
    targets = {n: dict(model.named_modules())[n] for n in pick}
    for n, m in targets.items():
        logging.info(f'Target {n}: weight shape={tuple(m.weight.shape)}, '
                     f'weight.requires_grad={m.weight.requires_grad}')

    # Build a SMALL calib batch
    g = TinyImageNetLoaderGenerator(val_batch_size=4, num_workers=0)
    loader = g.calib_loader(num=4, batch_size=4, seed=0)
    (inp, _) = next(iter(loader))
    inp = inp.to(device)
    logging.info(f'Input batch: {tuple(inp.shape)}')

    # Accumulators
    x_sq_acc = {n: None for n in pick}
    g_abs_acc = {n: None for n in pick}

    def make_fwd(name):
        def h(module, inputs, output):
            x = inputs[0].detach()
            x_sq = (x.float() ** 2).reshape(-1, x.shape[-1]).mean(dim=0)
            logging.info(f'  [FWD] {name}: x shape={tuple(x.shape)}, '
                         f'x_sq mean={x_sq.mean().item():.4e}, '
                         f'min={x_sq.min().item():.4e}, max={x_sq.max().item():.4e}')
            x_sq_acc[name] = x_sq
        return h

    def make_bwd(name):
        def h(module, grad_input, grad_output):
            g = grad_output[0]
            if g is None:
                logging.warning(f'  [BWD] {name}: grad_output[0] is None!')
                return
            g_abs = g.detach().float().abs().reshape(-1, g.shape[-1]).mean(dim=0)
            logging.info(f'  [BWD] {name}: grad shape={tuple(g.shape)}, '
                         f'|g| mean={g_abs.mean().item():.4e}, '
                         f'min={g_abs.min().item():.4e}, max={g_abs.max().item():.4e}')
            g_abs_acc[name] = g_abs
        return h

    hooks = []
    for n, m in targets.items():
        hooks.append(m.register_forward_hook(make_fwd(n)))
        hooks.append(m.register_full_backward_hook(make_bwd(n)))

    # Forward FP (get target softmax)
    T = 20.0
    with torch.no_grad():
        raw_pred = model(inp) / T
        raw_softmax = F.softmax(raw_pred, dim=-1)
    logging.info(f'FP pred shape={tuple(raw_pred.shape)}, sum check={raw_softmax.sum(dim=-1)[0].item():.4f}')

    # Now: forward with grad enabled, backward to get gradients into the Linears
    # Note: since we're using same FP model both times, KL = 0 → grad = 0!
    # Need to inject some perturbation. Use a different temperature.
    logging.info('Forward+backward with perturbed temperature (T=1 vs T=20 for target) ...')
    model.zero_grad(set_to_none=True)
    pred = model(inp) / 1.0  # ← different scale → different softmax
    loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_softmax, reduction='batchmean')
    logging.info(f'KL loss = {loss.item():.4f}')
    loss.backward()

    for h in hooks: h.remove()

    # Compute FIM(W) = |g| ⊗ x²
    logging.info('=== FIM(W) computation ===')
    for n in pick:
        x_sq = x_sq_acc[n]; g_abs = g_abs_acc[n]
        if x_sq is None or g_abs is None:
            logging.warning(f'{n}: missing stats (x_sq={x_sq is None}, g_abs={g_abs is None})')
            continue
        fim = g_abs.unsqueeze(1) * x_sq.unsqueeze(0)
        logging.info(f'{n}: fim shape={tuple(fim.shape)} | '
                     f'mean={fim.mean().item():.4e} | min={fim.min().item():.4e} | '
                     f'max={fim.max().item():.4e}')

    logging.info('Done.')


if __name__ == '__main__':
    main()
