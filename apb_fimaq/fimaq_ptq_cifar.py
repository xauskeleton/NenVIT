"""
Standalone FIMA-Q PTQ on CIFAR-100, reusing FIMA-Q/ utils UNMODIFIED. Model via --model
(default Swin-S; FIMA-Q wrap also supports the ViT/DeiT families with a matching --ckpt).

Per project rule (keep FIMA-Q/ pristine): this wrapper lives in apb_fimaq/ and only
*imports* FIMA-Q code. It swaps FIMA-Q's hardcoded ImageNet loader for a CIFAR-100
loader (32->224 bicubic, ImageNet-norm — same transform as our finetune/qat pipeline)
and loads the CIFAR-100 finetuned Swin-S baseline (ckpt/best_swin.pth, 90.88%).

This is the PTQ-only baseline (no QAT, no APB) to compare against APB+QAT at low bits.

Usage:
    python apb_fimaq/fimaq_ptq_cifar.py --w-bit 2 --a-bit 2            # calibrate-only (fast)
    python apb_fimaq/fimaq_ptq_cifar.py --w-bit 2 --a-bit 2 --optimize # + Fisher-DPLR block recon (real FIMA-Q)
"""
import os
import sys
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
import timm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'FIMA-Q'))          # utils.*, quant_layers.*

from utils.datasets import LoaderGenerator         # noqa: E402
from utils.calibrator import QuantCalibrator       # noqa: E402
from utils.block_recon import BlockReconstructor   # noqa: E402
from utils.wrap_net import (                        # noqa: E402
    wrap_modules_in_net, wrap_reparamed_modules_in_net)
from utils.test_utils import validate              # noqa: E402

from torchvision import transforms                 # noqa: E402
from torchvision.datasets import CIFAR100          # noqa: E402

CKPT = ROOT / 'ckpt' / 'best_swin.pth'
DATA_DIR = ROOT / 'data'


class Config:
    """Mirror of FIMA-Q configs/3bit/best.py, with bits parameterised."""
    def __init__(self, w_bit, a_bit, calib_size, optim_size, optim_batch_size):
        # calibration / optimization sizes
        self.calib_size = calib_size
        self.optim_size = optim_size
        self.calib_batch_size = 32
        self.optim_batch_size = optim_batch_size
        # bits
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.qconv_a_bit = 8          # patch-embed conv input kept 8-bit (as paper config)
        self.qhead_a_bit = a_bit      # classifier head act bits = a_bit
        # calibration
        self.calib_metric = 'mse'
        self.matmul_head_channel_wise = True
        self.token_channel_wise = True
        self.eq_n = 128
        self.search_round = 3
        # optimization (Fisher-DPLR block reconstruction — the FIMA-Q contribution)
        self.keep_gpu = True
        self.optim_metric = 'fisher_dplr'
        self.temp = 20
        self.k = 5
        self.p1 = 1.0
        self.p2 = 1.0
        self.dis_mode = 'q'
        self.optim_mode = 'qdrop'
        self.drop_prob = 0.5


class CifarLoaderFimaq(LoaderGenerator):
    """CIFAR-100 loader matching FIMA-Q's LoaderGenerator interface.

    Uses val (center, no-aug) transform for BOTH val and calib so PTQ calibration
    is deterministic (no random crop/flip noise) — standard for PTQ.
    """
    def load(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
        bicubic = transforms.InterpolationMode.BICUBIC
        self.val_transform = transforms.Compose([
            transforms.Resize(224, interpolation=bicubic),
            transforms.ToTensor(),
            normalize,
        ])
        self.train_transform = self.val_transform   # PTQ: no augmentation

    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = CIFAR100(root=str(DATA_DIR), train=True,
                                       download=True, transform=self.train_transform)
        return self._train_set

    @property
    def val_set(self):
        if self._val_set is None:
            self._val_set = CIFAR100(root=str(DATA_DIR), train=False,
                                     download=True, transform=self.val_transform)
        return self._val_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='swin_small_patch4_window7_224',
                   help='timm model (default swin_small_patch4_window7_224). FIMA-Q '
                        'wrap supports the Swin/ViT/DeiT families; use a matching '
                        '--ckpt baseline.')
    p.add_argument('--ckpt', type=str, default=str(CKPT),
                   help='Finetuned FP baseline state_dict for --model (default '
                        'ckpt/best_swin.pth, the Swin-S CIFAR-100 90.88% baseline).')
    p.add_argument('--w-bit', type=int, default=2)
    p.add_argument('--a-bit', type=int, default=2)
    p.add_argument('--optimize', action='store_true',
                   help='Run Fisher-DPLR block reconstruction (the real FIMA-Q step). '
                        'Without it: calibration-only (much faster, but low accuracy).')
    p.add_argument('--calib-size', type=int, default=128)
    p.add_argument('--optim-size', type=int, default=1024)
    p.add_argument('--optim-batch-size', type=int, default=32)
    p.add_argument('--val-batch-size', type=int, default=200)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--no-keep-gpu', action='store_true',
                   help='Cache block I/O on CPU during optimize (slower, less VRAM).')
    p.add_argument('--skip-fp-eval', action='store_true',
                   help='Skip the full-precision sanity eval (saves ~1-2 min).')
    p.add_argument('--seed', type=int, default=3407)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print(f'FIMA-Q PTQ on {args.model} | CIFAR-100 | W{args.w_bit}/A{args.a_bit} | device={device}')
    print(f'optimize(Fisher-DPLR)={args.optimize} | calib={args.calib_size} '
          f'optim={args.optim_size} | seed={args.seed}')
    print('=' * 60)

    # ----- model: CIFAR-100 finetuned baseline (default Swin-S, 90.88%) -----
    ckpt = Path(args.ckpt)
    model = timm.create_model(args.model, pretrained=True, num_classes=100)
    assert ckpt.exists(), f'Missing baseline checkpoint: {ckpt}'
    sd = torch.load(ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    model.load_state_dict(sd, strict=True)
    print(f'>>> Loaded CIFAR-100 baseline: {ckpt}')
    model.to(device).eval()

    cfg = Config(args.w_bit, args.a_bit, args.calib_size,
                 args.optim_size, args.optim_batch_size)
    if args.no_keep_gpu:
        cfg.keep_gpu = False

    g = CifarLoaderFimaq(root=str(DATA_DIR), val_batch_size=args.val_batch_size,
                         num_workers=args.num_workers)
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(device)

    if not args.skip_fp_eval:
        print('\n[FP baseline] eval (sanity, expect ~90.88%) ...')
        validate(val_loader, model, criterion, device=device)

    # ----- wrap + calibrate (W{w}/A{a}) -----
    reparam = True
    print(f'\nWrapping quant modules (reparam={reparam}) ...')
    model = wrap_modules_in_net(model, cfg, reparam=reparam)
    model.to(device).eval()

    print(f'\nCalibrating ({cfg.calib_metric}, calib_size={cfg.calib_size}) ...')
    calib_loader = g.calib_loader(num=cfg.calib_size,
                                  batch_size=cfg.calib_batch_size, seed=args.seed)
    QuantCalibrator(model, calib_loader).batching_quant_calib()
    model = wrap_reparamed_modules_in_net(model)
    model.to(device)

    print(f'\n[After calibration] W{args.w_bit}/A{args.a_bit} (calib-only, no block recon):')
    validate(val_loader, model, criterion, device=device)

    # ----- optional: Fisher-DPLR block reconstruction (real FIMA-Q) -----
    if args.optimize:
        print(f'\nFisher-DPLR block reconstruction '
              f'(optim_size={cfg.optim_size}, keep_gpu={cfg.keep_gpu}) ...')
        t0 = time.time()
        calib_loader = g.calib_loader(num=cfg.optim_size,
                                      batch_size=cfg.optim_batch_size, seed=args.seed)
        br = BlockReconstructor(model, cfg.optim_batch_size, calib_loader,
                                metric=cfg.optim_metric, temp=cfg.temp, k=cfg.k,
                                dis_mode=cfg.dis_mode, p1=cfg.p1, p2=cfg.p2)
        br.reconstruct_model(quant_act=True, mode=cfg.optim_mode,
                             drop_prob=cfg.drop_prob, keep_gpu=cfg.keep_gpu)
        print(f'Block reconstruction done in {time.time() - t0:.1f}s')
        print(f'\n[After FIMA-Q optimize] W{args.w_bit}/A{args.a_bit} (fisher_dplr):')
        validate(val_loader, model, criterion, device=device)

    print('\n' + '=' * 60)
    print('DONE.')
    print('=' * 60)


if __name__ == '__main__':
    main()
