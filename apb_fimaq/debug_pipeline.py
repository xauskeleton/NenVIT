"""
Fast end-to-end debug for APB+FIMA-Q pipeline.

Sizes scaled down so the full A→B→C runs in 5-10 minutes:
  - calib_size=8 (just enough for scale search)
  - eval on 5 val batches only (~320 samples)
  - skip Phase A reconstruction (big speedup, baseline will be poor but that's fine)
  - skip Phase D (known wrap_quantizers_in_net incompat)

What we want to verify:
  [ ] Phase A wrap + calib runs and produces valid quantized model
  [ ] Phase B FIM extraction runs, output shapes match weight shapes
  [ ] FIM distribution sanity check (not all zero, has structure)
  [ ] Phase C replace_weight_quantizers_with_apb runs
  [ ] Model still forward-passes after APB swap
  [ ] Accuracy drop is sensible (not 0%, not random)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / 'apb_fimaq', PROJECT_ROOT / 'FIMA-Q' / 'FIMA-Q',
          PROJECT_ROOT / 'scripts', PROJECT_ROOT):
    sys.path.insert(0, str(p))

import timm  # noqa: E402
from utils.calibrator import QuantCalibrator  # noqa: E402
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net  # noqa: E402

from tiny_imagenet_loader import TinyImageNetLoaderGenerator  # noqa: E402
from fim_weight_extractor import FIMWeightExtractor  # noqa: E402
from apb_wrap import get_apb_target_modules, replace_weight_quantizers_with_apb, log_apb_stats  # noqa: E402


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S', force=True)


def quick_eval(model, val_loader, device, max_batches=5):
    """Fast eval on first few batches."""
    model.eval()
    correct1 = correct5 = total = 0
    crit = nn.CrossEntropyLoss().to(device)
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches: break
            x = x.to(device); y = y.to(device)
            out = model(x)
            losses.append(crit(out, y).item())
            _, pred5 = out.topk(5, dim=1)
            correct1 += (pred5[:, 0] == y).sum().item()
            correct5 += (pred5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)
    return correct1/total * 100, correct5/total * 100, sum(losses)/len(losses), total


def make_config():
    cfg = argparse.Namespace()
    cfg.w_bit = 4; cfg.a_bit = 4
    cfg.qconv_a_bit = 8; cfg.qhead_a_bit = 4
    cfg.calib_metric = 'mse'
    cfg.calib_batch_size = 8
    cfg.matmul_head_channel_wise = True; cfg.token_channel_wise = True
    cfg.eq_n = 32  # ← drastically reduce from 128 for debug speed
    cfg.search_round = 1  # ← from 3
    cfg.temp = 20
    return cfg


def main():
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    logging.info('[setup] Loading Swin-S pretrained ...')
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    model.to(device).eval()

    logging.info('[setup] Building Tiny ImageNet loaders ...')
    g = TinyImageNetLoaderGenerator(val_batch_size=64, num_workers=0)
    val_loader = g.val_loader()
    cfg = make_config()

    # ------------------------------------------------------------------
    # Baseline FP
    # ------------------------------------------------------------------
    t0 = time.time()
    top1, top5, loss, n = quick_eval(model, val_loader, device)
    logging.info(f'[FP baseline] top1={top1:.2f}% top5={top5:.2f}% loss={loss:.3f} '
                 f'({n} samples, {time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # PHASE A: wrap + calibrate (fast, no reconstruction)
    # ------------------------------------------------------------------
    logging.info('[Phase A] Wrapping with FIMA-Q quant layers (reparam=True) ...')
    model = wrap_modules_in_net(model, cfg, reparam=True)
    model.to(device).eval()

    logging.info('[Phase A] Calibration (8 imgs, eq_n=32, search=1) ...')
    calib_loader = g.calib_loader(num=8, batch_size=8, seed=42)
    t0 = time.time()
    QuantCalibrator(model, calib_loader).batching_quant_calib()
    logging.info(f'[Phase A] Calibration done in {time.time()-t0:.1f}s')

    model = wrap_reparamed_modules_in_net(model)
    model.to(device).eval()

    top1, top5, loss, n = quick_eval(model, val_loader, device)
    logging.info(f'[Phase A] After calib: top1={top1:.2f}% top5={top5:.2f}% loss={loss:.3f}')

    # ------------------------------------------------------------------
    # PHASE B: extract F_diag(W)
    # ------------------------------------------------------------------
    logging.info('[Phase B] Identifying APB targets ...')
    targets = get_apb_target_modules(model)
    logging.info(f'[Phase B] Found {len(targets)} APB targets')
    for name in list(targets.keys())[:3]:
        m = targets[name]
        logging.info(f'         e.g. {name}: weight={tuple(m.weight.shape)}, '
                     f'n_V={getattr(m, "n_V", 1)}, '
                     f'mode={getattr(m, "mode", "?")}')

    logging.info('[Phase B] Extracting F_diag(W) with KL-divergence forward+backward ...')
    fim_loader = g.calib_loader(num=8, batch_size=8, seed=43)
    extractor = FIMWeightExtractor(model, targets, fim_loader, temperature=cfg.temp)
    t0 = time.time()
    try:
        extractor.run(device=device)
    except Exception as e:
        logging.error(f'[Phase B] FAILED during run(): {type(e).__name__}: {e}')
        raise
    logging.info(f'[Phase B] FIM extraction done in {time.time()-t0:.1f}s '
                 f'({extractor.n_samples} samples)')

    # Sanity check FIM
    fim_dict = extractor.all_fim()
    logging.info(f'[Phase B] Got FIM for {len(fim_dict)} layers')
    sample = list(fim_dict.items())[:3]
    for name, fim in sample:
        target_w = targets[name].weight
        logging.info(f'         {name}: fim shape={tuple(fim.shape)} '
                     f'(weight shape={tuple(target_w.shape)}) '
                     f'mean={fim.mean().item():.3e} '
                     f'min={fim.min().item():.3e} max={fim.max().item():.3e}')

    # Verify all FIMs have valid stats (non-zero, finite)
    zero_layers = [n for n, f in fim_dict.items() if f.abs().max() < 1e-12]
    nan_layers = [n for n, f in fim_dict.items() if not torch.isfinite(f).all()]
    logging.info(f'[Phase B] Sanity: {len(zero_layers)} all-zero layers, '
                 f'{len(nan_layers)} layers with NaN/Inf')
    if zero_layers:
        logging.warning(f'         ZERO LAYERS (first 3): {zero_layers[:3]}')

    # ------------------------------------------------------------------
    # PHASE C: replace w_quantizer with APBWeightQuantizer
    # ------------------------------------------------------------------
    logging.info('[Phase C] Replacing w_quantizers with APB (binary_ratio=0.75) ...')
    try:
        stats = replace_weight_quantizers_with_apb(model, fim_dict, binary_ratio=0.75)
    except Exception as e:
        logging.error(f'[Phase C] FAILED: {type(e).__name__}: {e}')
        raise
    log_apb_stats(stats)

    # Ensure quant_forward mode for eval
    for m in model.modules():
        if hasattr(m, 'mode'): m.mode = 'quant_forward'

    # ------------------------------------------------------------------
    # Eval AFTER APB partition (no Phase D fine-tune)
    # ------------------------------------------------------------------
    logging.info('[eval] Forward pass after APB swap ...')
    try:
        top1, top5, loss, n = quick_eval(model, val_loader, device)
    except Exception as e:
        logging.error(f'[eval] Forward FAILED after APB: {type(e).__name__}: {e}')
        raise
    logging.info(f'[Phase C eval] top1={top1:.2f}% top5={top5:.2f}% loss={loss:.3f}')

    logging.info('=== DEBUG PIPELINE PASSED ===')
    logging.info('All phases A→B→C executed without crash.')
    logging.info('Phase D skipped (known incompat with FIMA-Q wrap_quantizers_in_net).')


if __name__ == '__main__':
    main()
