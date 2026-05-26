"""
Full APB + FIMA-Q pipeline on Tiny ImageNet.

Phases:
  A. Standard FIMA-Q calibration + block reconstruction (W4A4 baseline)
  B. Extract F_diag(W) per APB target Linear (96 layers)
  C. Replace AdaRound w_quantizer → APBWeightQuantizer with FIM-driven mask
  D. Re-run block reconstruction with APB weights (DPLR-FIM loss unchanged)
  E. Evaluate

Does NOT modify FIMA-Q repo. Imports its modules and subclasses where needed.

Usage:
  python run_apb_pipeline.py                          # default 75% binary all roles
  python run_apb_pipeline.py --binary-ratio 0.85      # more aggressive
  python run_apb_pipeline.py --skip-phase-a-recon     # use already-calibrated ckpt
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIMA_Q_ROOT = PROJECT_ROOT / 'FIMA-Q' / 'FIMA-Q'
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'
APB_ROOT = PROJECT_ROOT / 'apb_fimaq'
for p in (str(APB_ROOT), str(FIMA_Q_ROOT), str(SCRIPTS_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# FIMA-Q (untouched)
import timm  # noqa: E402
from utils.calibrator import QuantCalibrator  # noqa: E402
from utils.block_recon import BlockReconstructor  # noqa: E402
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net  # noqa: E402
from utils.test_utils import validate  # noqa: E402

# Our APB code
from tiny_imagenet_loader import TinyImageNetLoaderGenerator  # noqa: E402
from fim_weight_extractor import FIMWeightExtractor  # noqa: E402
from apb_wrap import get_apb_target_modules, replace_weight_quantizers_with_apb, log_apb_stats  # noqa: E402


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'apb_pipeline_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    return log_path


def make_config(w_bit=4, a_bit=4, calib_size=64, optim_size=256, optim_batch_size=16):
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


def evaluate(model, loader, device, label=''):
    crit = nn.CrossEntropyLoss().to(device)
    logging.info(f'=== Evaluating {label} ===')
    return validate(loader, model, crit, print_freq=20, device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--w-bit', type=int, default=4)
    p.add_argument('--a-bit', type=int, default=4)
    p.add_argument('--calib-size', type=int, default=64)
    p.add_argument('--optim-size', type=int, default=256)
    p.add_argument('--optim-batch-size', type=int, default=16)
    p.add_argument('--val-batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=3407)

    # APB-specific knobs
    p.add_argument('--binary-ratio', type=float, default=0.75,
                   help='default target binary ratio per layer (0.0=none, 1.0=all binary)')
    p.add_argument('--ratio-qkv',  type=float, default=None, help='binary ratio for qkv layers')
    p.add_argument('--ratio-proj', type=float, default=None, help='binary ratio for attn.proj')
    p.add_argument('--ratio-fc1',  type=float, default=None, help='binary ratio for mlp.fc1')
    p.add_argument('--ratio-fc2',  type=float, default=None, help='binary ratio for mlp.fc2')

    p.add_argument('--load-calibrate-ckpt', type=str, default=None,
                   help='skip Phase A calibration, load from this checkpoint')
    p.add_argument('--load-optimize-ckpt', type=str, default=None,
                   help='skip Phase A reconstruction, load from this checkpoint')
    p.add_argument('--skip-phase-d', action='store_true',
                   help='skip final APB reconstruction (just measure post-APB eval)')

    p.add_argument('--out-dir', type=str,
                   default=str(PROJECT_ROOT / 'checkpoints' / 'apb_pipeline'))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    log_path = setup_logging(out_dir)
    logging.info(f'Log: {log_path}')
    logging.info(f'Args: {vars(args)}')

    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # 0. Model + dataloaders
    # ------------------------------------------------------------------
    logging.info('Loading Swin-S pretrained ...')
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    model.to(device).eval()

    logging.info('Building Tiny ImageNet loaders ...')
    g = TinyImageNetLoaderGenerator(val_batch_size=args.val_batch_size,
                                    num_workers=args.num_workers)
    val_loader = g.val_loader()
    cfg = make_config(args.w_bit, args.a_bit, args.calib_size,
                      args.optim_size, args.optim_batch_size)

    # ------------------------------------------------------------------
    # PHASE A: FIMA-Q W4A4 calibration + block reconstruction
    # ------------------------------------------------------------------
    logging.info('========== PHASE A: FIMA-Q baseline ==========')
    model = wrap_modules_in_net(model, cfg, reparam=(args.load_calibrate_ckpt is None
                                                      and args.load_optimize_ckpt is None))
    model.to(device).eval()

    if args.load_optimize_ckpt:
        logging.info(f'Loading optimized checkpoint: {args.load_optimize_ckpt}')
        ckpt = torch.load(args.load_optimize_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        for m in model.modules():
            if hasattr(m, 'mode'): m.mode = 'quant_forward'
            if hasattr(m, 'calibrated'): m.calibrated = True
            for attr in ('a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer'):
                if hasattr(m, attr): getattr(m, attr).inited = True
    elif args.load_calibrate_ckpt:
        logging.info(f'Loading calibrated checkpoint: {args.load_calibrate_ckpt}')
        ckpt = torch.load(args.load_calibrate_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        for m in model.modules():
            if hasattr(m, 'mode'): m.mode = 'quant_forward'
            if hasattr(m, 'calibrated'): m.calibrated = True
            for attr in ('a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer'):
                if hasattr(m, attr): getattr(m, attr).inited = True
        # Re-run reconstruction to bake AdaRound
        _run_phase_a_reconstruction(model, g, cfg, args, device, out_dir)
    else:
        # Fresh calibration + reconstruction
        logging.info('--- Calibration (scale search) ---')
        calib_loader = g.calib_loader(num=cfg.calib_size,
                                      batch_size=cfg.calib_batch_size, seed=args.seed)
        QuantCalibrator(model, calib_loader).batching_quant_calib()
        model = wrap_reparamed_modules_in_net(model)
        model.to(device).eval()
        ckpt_calib = out_dir / 'phase_a_calibrated.pth'
        torch.save(model.state_dict(), ckpt_calib)
        logging.info(f'Saved: {ckpt_calib}')
        evaluate(model, val_loader, device, 'after Phase A calibration')

        _run_phase_a_reconstruction(model, g, cfg, args, device, out_dir)

    evaluate(model, val_loader, device, 'after Phase A (FIMA-Q W4A4 baseline)')

    # ------------------------------------------------------------------
    # PHASE B: Extract F_diag(W) per APB target
    # ------------------------------------------------------------------
    logging.info('========== PHASE B: Extract F_diag(W) ==========')
    targets = get_apb_target_modules(model)
    logging.info(f'Found {len(targets)} APB target Linears')

    fim_loader = g.calib_loader(num=cfg.calib_size,
                                batch_size=cfg.calib_batch_size, seed=args.seed)
    extractor = FIMWeightExtractor(model, targets, fim_loader, temperature=cfg.temp)
    t0 = time.time()
    extractor.run(device=device)
    fim_dict = extractor.all_fim()
    logging.info(f'F_diag(W) extracted for {len(fim_dict)} layers in {time.time()-t0:.1f}s')

    fim_path = out_dir / 'fim_weight.pt'
    torch.save(fim_dict, fim_path)
    logging.info(f'Saved FIM dict: {fim_path}')

    # ------------------------------------------------------------------
    # PHASE C: Replace w_quantizer with APBWeightQuantizer
    # ------------------------------------------------------------------
    logging.info('========== PHASE C: Apply APB (replace w_quantizer + set mask) ==========')
    per_role = {}
    if args.ratio_qkv  is not None: per_role['qkv']  = args.ratio_qkv
    if args.ratio_proj is not None: per_role['proj'] = args.ratio_proj
    if args.ratio_fc1  is not None: per_role['fc1']  = args.ratio_fc1
    if args.ratio_fc2  is not None: per_role['fc2']  = args.ratio_fc2
    logging.info(f'Binary ratio: default={args.binary_ratio}, per_role={per_role}')

    stats = replace_weight_quantizers_with_apb(model, fim_dict,
                                                binary_ratio=args.binary_ratio,
                                                per_role_ratio=per_role)
    log_apb_stats(stats)

    # Restore mode to quant_forward (replacement keeps the same flow)
    for m in model.modules():
        if hasattr(m, 'mode'): m.mode = 'quant_forward'

    evaluate(model, val_loader, device, 'AFTER APB partition (no re-reconstruction yet)')

    if args.skip_phase_d:
        logging.info('--skip-phase-d set; exiting after partition-only eval.')
        return

    # ------------------------------------------------------------------
    # PHASE D: Re-run block reconstruction with APB weights
    # ------------------------------------------------------------------
    logging.info('========== PHASE D: Re-reconstruct with APB weights ==========')
    optim_loader = g.calib_loader(num=cfg.optim_size,
                                  batch_size=cfg.optim_batch_size, seed=args.seed)
    t0 = time.time()
    recon = BlockReconstructor(model, cfg.optim_batch_size, optim_loader,
                               metric=cfg.optim_metric, temp=cfg.temp,
                               k=cfg.k, dis_mode=cfg.dis_mode,
                               p1=cfg.p1, p2=cfg.p2)
    # Note: BlockReconstructor.wrap_quantizers_in_net would normally REPLACE w_quantizer
    # with AdaRound. APBWeightQuantizer is incompatible (no .alpha as learnable). We
    # skip wrap_quantizers_in_net for APB layers by patching their w_quantizer to not
    # be MinMaxQuantLinear's UniformQuantizer (APBWeightQuantizer doesn't satisfy
    # isinstance check, so wrap_quantizers_in_net skips it cleanly).
    recon.reconstruct_model(quant_act=True, mode=cfg.optim_mode,
                            drop_prob=cfg.drop_prob, keep_gpu=cfg.keep_gpu)
    logging.info(f'Phase D reconstruction done in {(time.time()-t0)/60:.1f} min')

    ckpt_apb = out_dir / f'apb_w{cfg.w_bit}a{cfg.a_bit}_br{args.binary_ratio}.pth'
    torch.save(model.state_dict(), ckpt_apb)
    logging.info(f'Saved APB checkpoint: {ckpt_apb}')

    evaluate(model, val_loader, device, 'AFTER APB + Phase D re-reconstruction')

    logging.info('Pipeline complete.')


def _run_phase_a_reconstruction(model, g, cfg, args, device, out_dir):
    logging.info('--- DPLR-FIM block reconstruction (Phase A) ---')
    optim_loader = g.calib_loader(num=cfg.optim_size,
                                  batch_size=cfg.optim_batch_size, seed=args.seed)
    t0 = time.time()
    recon = BlockReconstructor(model, cfg.optim_batch_size, optim_loader,
                               metric=cfg.optim_metric, temp=cfg.temp,
                               k=cfg.k, dis_mode=cfg.dis_mode,
                               p1=cfg.p1, p2=cfg.p2)
    recon.reconstruct_model(quant_act=True, mode=cfg.optim_mode,
                            drop_prob=cfg.drop_prob, keep_gpu=cfg.keep_gpu)
    logging.info(f'Phase A reconstruction done in {(time.time()-t0)/60:.1f} min')
    ckpt_opt = out_dir / 'phase_a_optimized.pth'
    torch.save(model.state_dict(), ckpt_opt)
    logging.info(f'Saved: {ckpt_opt}')


if __name__ == '__main__':
    main()
