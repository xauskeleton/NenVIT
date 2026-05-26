"""
Phase A — FIMA-Q W4A4 baseline on Tiny ImageNet (no APB yet).

This script runs the standard FIMA-Q pipeline using our TinyImageNetLoaderGenerator:
  1. Load pretrained Swin-S from timm
  2. Wrap Linear/Conv2d/MatMul with FIMA-Q quant layers
  3. Calibrate quantizer scales via QuantCalibrator
  4. Block-wise reconstruction with DPLR-FIM loss
  5. Evaluate top-1/top-5 on Tiny val (182 classes mapped to ImageNet-1k)

Does NOT modify any file in FIMA-Q/. Imports its modules through sys.path.

Outputs:
  - checkpoints/phase_a/swin_small_w4a4_calibrated.pth
  - checkpoints/phase_a/swin_small_w4a4_optimized.pth
  - logs to stdout + file
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# --- Make FIMA-Q importable WITHOUT modifying it ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIMA_Q_ROOT = PROJECT_ROOT / 'FIMA-Q' / 'FIMA-Q'
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'
sys.path.insert(0, str(FIMA_Q_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# FIMA-Q imports (untouched original code)
import timm  # noqa: E402
from utils.calibrator import QuantCalibrator  # noqa: E402
from utils.block_recon import BlockReconstructor  # noqa: E402
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net  # noqa: E402
from utils.test_utils import validate  # noqa: E402

# Our code
from tiny_imagenet_loader import TinyImageNetLoaderGenerator  # noqa: E402


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'phase_a_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    return log_path


def make_config(w_bit=4, a_bit=4, calib_size=64, optim_size=256, optim_batch_size=16):
    """Mini Config matching FIMA-Q's Config interface (configs/4bit/best.py),
    but sized down for Tiny ImageNet debug runs."""
    cfg = argparse.Namespace()
    cfg.optim_size = optim_size
    cfg.calib_size = calib_size
    cfg.optim_batch_size = optim_batch_size
    cfg.calib_batch_size = 16
    cfg.w_bit = w_bit
    cfg.a_bit = a_bit
    cfg.qconv_a_bit = 8
    cfg.qhead_a_bit = a_bit
    cfg.calib_metric = 'mse'
    cfg.matmul_head_channel_wise = True
    cfg.token_channel_wise = True
    cfg.eq_n = 128
    cfg.search_round = 3
    cfg.keep_gpu = True
    cfg.optim_metric = 'fisher_dplr'
    cfg.temp = 20
    cfg.k = 5
    cfg.p1 = 1.0
    cfg.p2 = 1.0
    cfg.dis_mode = 'q'
    cfg.optim_mode = 'qdrop'
    cfg.drop_prob = 0.5
    return cfg


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    crit = nn.CrossEntropyLoss().to(device)
    return validate(loader, model, crit, print_freq=20, device=device)


def main():
    parser = argparse.ArgumentParser(description='Phase A: FIMA-Q W4A4 on Tiny ImageNet')
    parser.add_argument('--w-bit', type=int, default=4)
    parser.add_argument('--a-bit', type=int, default=4)
    parser.add_argument('--calib-size', type=int, default=64,
                        help='# images for scale search (default 64 for Tiny debug)')
    parser.add_argument('--optim-size', type=int, default=256,
                        help='# images for DPLR-FIM block reconstruction (default 256)')
    parser.add_argument('--optim-batch-size', type=int, default=16)
    parser.add_argument('--val-batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--skip-eval-fp', action='store_true',
                        help='skip FP baseline evaluation (saves 1 min)')
    parser.add_argument('--skip-optimize', action='store_true',
                        help='only run calibration, skip DPLR-FIM block reconstruction')
    parser.add_argument('--out-dir', type=str,
                        default=str(PROJECT_ROOT / 'checkpoints' / 'phase_a'))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    log_path = setup_logging(out_dir)
    logging.info(f'Log file: {log_path}')
    logging.info(f'Args: {vars(args)}')

    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    # ------------------------------------------------------------------
    # 1. Load pretrained Swin-S
    # ------------------------------------------------------------------
    logging.info('Loading pretrained Swin-S from timm ...')
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f'Model loaded: swin_small_patch4_window7_224 ({n_params:.2f}M params)')

    # ------------------------------------------------------------------
    # 2. Tiny ImageNet dataloaders
    # ------------------------------------------------------------------
    logging.info('Building Tiny ImageNet loaders ...')
    g = TinyImageNetLoaderGenerator(val_batch_size=args.val_batch_size,
                                    num_workers=args.num_workers,
                                    remap_to_imagenet1k=True,
                                    drop_unmappable=True)
    val_loader = g.val_loader()
    logging.info(f'Val: {len(g.val_set)} samples ({len(val_loader)} batches)')

    # ------------------------------------------------------------------
    # 3. (optional) Evaluate FP baseline
    # ------------------------------------------------------------------
    if not args.skip_eval_fp:
        logging.info('=== Evaluating FP baseline ===')
        evaluate(model, val_loader, device)

    # ------------------------------------------------------------------
    # 4. Config + Wrap quant layers (uses FIMA-Q's wrap_modules_in_net untouched)
    # ------------------------------------------------------------------
    cfg = make_config(w_bit=args.w_bit, a_bit=args.a_bit,
                      calib_size=args.calib_size, optim_size=args.optim_size,
                      optim_batch_size=args.optim_batch_size)
    logging.info(f'Config: w{cfg.w_bit}a{cfg.a_bit}, calib={cfg.calib_size}, '
                 f'optim={cfg.optim_size}, k={cfg.k}, optim_metric={cfg.optim_metric}')

    logging.info('Wrapping Linear/Conv2d/MatMul with FIMA-Q quant layers (reparam=True) ...')
    model = wrap_modules_in_net(model, cfg, reparam=True)
    model.to(device).eval()

    # ------------------------------------------------------------------
    # 5. Calibration (find scales via brute-force search)
    # ------------------------------------------------------------------
    logging.info('=== Calibration (scale search) ===')
    calib_loader = g.calib_loader(num=cfg.calib_size,
                                  batch_size=cfg.calib_batch_size, seed=args.seed)
    t0 = time.time()
    QuantCalibrator(model, calib_loader).batching_quant_calib()
    logging.info(f'Calibration done in {time.time()-t0:.1f}s')

    logging.info('Reparam fusion ...')
    model = wrap_reparamed_modules_in_net(model)
    model.to(device).eval()

    ckpt_calib = out_dir / f'swin_small_w{cfg.w_bit}a{cfg.a_bit}_calibrated.pth'
    torch.save(model.state_dict(), ckpt_calib)
    logging.info(f'Saved calibrated checkpoint: {ckpt_calib}')

    logging.info('=== Evaluating after calibration only (no AdaRound yet) ===')
    evaluate(model, val_loader, device)

    if args.skip_optimize:
        logging.info('--skip-optimize set, exiting after calibration.')
        return

    # ------------------------------------------------------------------
    # 6. Block-wise reconstruction with DPLR-FIM (the heart of FIMA-Q)
    # ------------------------------------------------------------------
    logging.info('=== DPLR-FIM block reconstruction ===')
    optim_loader = g.calib_loader(num=cfg.optim_size,
                                  batch_size=cfg.optim_batch_size, seed=args.seed)
    t0 = time.time()
    recon = BlockReconstructor(model, cfg.optim_batch_size, optim_loader,
                               metric=cfg.optim_metric, temp=cfg.temp,
                               k=cfg.k, dis_mode=cfg.dis_mode,
                               p1=cfg.p1, p2=cfg.p2)
    recon.reconstruct_model(quant_act=True, mode=cfg.optim_mode,
                            drop_prob=cfg.drop_prob, keep_gpu=cfg.keep_gpu)
    logging.info(f'Block reconstruction done in {(time.time()-t0)/60:.1f} min')

    ckpt_opt = out_dir / f'swin_small_w{cfg.w_bit}a{cfg.a_bit}_optimized.pth'
    torch.save(model.state_dict(), ckpt_opt)
    logging.info(f'Saved optimized checkpoint: {ckpt_opt}')

    logging.info('=== Evaluating after DPLR-FIM reconstruction ===')
    evaluate(model, val_loader, device)

    logging.info('Phase A complete.')


if __name__ == '__main__':
    main()
