"""
FIMA-Q-guided structural pruning of Swin-S on CIFAR-100 (standalone, no quant).

Pipeline (matches the 'prune + finetune separately first' plan):
  1. load CIFAR-100 finetuned baseline (ckpt/best.pth, 90.88%)
  2. compute DPLR-FIM importance (FIMA-Q) on a few calib batches
  3. structurally prune heads + MLP neurons per block (lowest-FIM dropped)
  4. eval right after prune  -> 'prune cost' before recovery
  5. finetune on CIFAR-100   -> recover accuracy
  6. eval -> final pruned accuracy + param reduction

Quant (APB+FIMA-Q) is intentionally NOT applied here — this isolates pruning cost.
Combine with quant later by feeding the saved pruned best.pth into qat.py.

USAGE
=====
# global (recommended)
python apb_fimaq/prune_swin_cifar.py --rank-mode global --global-metric per_param --head-ratio 0.25 --mlp-ratio 0.25 --min-heads 1 --mlp-keep-frac 0.05 --fim-batches 10 --epochs 15 --lr 1e-4 --batch-size 64 --num-workers 4 --seed 3407

# per_layer
python apb_fimaq/prune_swin_cifar.py --rank-mode per_layer --head-ratio 0.25 --mlp-ratio 0.25 --fim-batches 10 --epochs 15 --lr 1e-4 --batch-size 64 --num-workers 4 --seed 3407

# quick smoke
python apb_fimaq/prune_swin_cifar.py --debug
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import timm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qat import (evaluate, _Tee, compute_weight_dplr_fim,        # noqa: E402
                 init_from_baseline)
from finetune import build_data                                  # noqa: E402
from prune import prune_swin                                     # noqa: E402

MODEL = 'swin_small_patch4_window7_224'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar100')
    p.add_argument('--data-dir', type=str, default='')
    p.add_argument('--head-ratio', type=float, default=0.25,
                   help='Fraction of attention heads to prune PER BLOCK.')
    p.add_argument('--mlp-ratio', type=float, default=0.25,
                   help='Fraction of MLP hidden neurons to prune PER BLOCK.')
    p.add_argument('--min-heads', type=int, default=1,
                   help='Never prune a block below this many heads.')
    p.add_argument('--rank-mode', choices=['per_layer', 'global'], default='per_layer',
                   help='per_layer: same %% pruned in each block. '
                        'global: pool units across blocks, prune globally lowest.')
    p.add_argument('--global-metric', choices=['per_param', 'raw'], default='per_param',
                   help='Global ranking score: per_param (importance/param-cost, '
                        'compression-aware) or raw (importance only).')
    p.add_argument('--head-keep-frac', type=float, default=0.0,
                   help='Global only: each block keeps >= this fraction of heads '
                        '(on top of --min-heads). Default 0 (floor = min-heads = 1).')
    p.add_argument('--mlp-keep-frac', type=float, default=0.05,
                   help='Global only: each block keeps >= this fraction of MLP '
                        'neurons (safety floor, avoids 1-neuron blocks). Default 0.05.')
    p.add_argument('--fim-batches', type=int, default=10,
                   help='Calib batches for DPLR-FIM importance.')
    p.add_argument('--fim-mode', choices=['dplr', 'rank', 'diag'], default='dplr')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--debug', action='store_true',
                   help='2 epochs, 6 train batches/epoch, tiny eval, fim-batches=2.')
    p.add_argument('--out-dir', type=str,
                   default=str(PROJECT_ROOT / 'ckpt' / 'pruned_cifar100'))
    p.add_argument('--log-file', type=str, default='')
    args = p.parse_args()

    if args.debug:
        args.epochs = 2
        args.batch_size = min(args.batch_size, 16)
        args.num_workers = 0
        args.fim_batches = 2

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_path = args.log_file or str(out_dir / 'train.log')
    if log_path.lower() != 'none':
        log_fp = open(log_path, 'a', buffering=1, encoding='utf-8')
        sys.stdout = _Tee(sys.stdout, log_fp)
        print(f'\n[{datetime.now():%Y-%m-%d %H:%M:%S}] ===== prune run '
              f'(log → {log_path}) =====')

    print('=' * 60)
    print(f'FIMA-Q-guided structural prune | Swin-S | device={device}')
    print(f'Args: {vars(args)}')
    print('=' * 60)

    train_loader, val_loader, num_classes = build_data(args)
    max_b = 10 if args.debug else None

    print(f'Loading Swin-S (num_classes={num_classes}) ...')
    model = timm.create_model(MODEL, pretrained=True, num_classes=num_classes)
    status = init_from_baseline(model)
    if status == 'loaded':
        print(f'>>> Initialised from CIFAR baseline: {PROJECT_ROOT/"ckpt"/"best.pth"}')
    else:
        print(f'>>> No/incompatible baseline ({status}) — using pretrained '
              f'(head random; numbers only relative).')
    model.to(device)

    t1, t5, n = evaluate(model, val_loader, device, max_batches=max_b)
    print(f'[FP baseline]   top1={t1:.2f}% top5={t5:.2f}% ({n} samples)')

    # ----- FIMA-Q importance -----
    print(f'Computing DPLR-FIM importance over {args.fim_batches} batches '
          f'(mode={args.fim_mode}) ...')
    t0 = time.time()
    fim = compute_weight_dplr_fim(model, train_loader, device,
                                  n_batches=args.fim_batches, fim_mode=args.fim_mode,
                                  scope='skip')
    print(f'FIM done in {time.time()-t0:.1f}s, {len(fim)} layers')

    # ----- structural prune -----
    rank_desc = (f'global/{args.global_metric}' if args.rank_mode == 'global'
                 else 'per_layer')
    print(f'Pruning: head_ratio={args.head_ratio}, mlp_ratio={args.mlp_ratio}, '
          f'rank={rank_desc} (lowest-FIM dropped) ...')
    stats = prune_swin(model, fim, head_ratio=args.head_ratio,
                       mlp_ratio=args.mlp_ratio, min_heads=args.min_heads,
                       rank_mode=args.rank_mode, global_metric=args.global_metric,
                       head_keep_frac=args.head_keep_frac,
                       mlp_keep_frac=args.mlp_keep_frac)
    model.to(device)
    print(f'  heads  {stats["heads_before"]} -> {stats["heads_after"]}')
    print(f'  mlp    {stats["mlp_neurons_before"]} -> {stats["mlp_neurons_after"]}')
    print(f'  params {stats["params_before"]/1e6:.2f}M -> '
          f'{stats["params_after"]/1e6:.2f}M ({stats["param_reduction"]:.1%} smaller)')

    t1, t5, n = evaluate(model, val_loader, device, max_batches=max_b)
    print(f'[Post-prune]    top1={t1:.2f}% top5={t5:.2f}% ({n}) - before finetune')

    # ----- finetune to recover -----
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = bool(args.amp) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print('=' * 60)
    print(f'Finetuning pruned model: {args.epochs} epochs, lr={args.lr}, amp={use_amp}')
    print('=' * 60)

    best_top1 = 0.0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time(); loss_sum = 0; loss_n = 0
        pbar = tqdm(train_loader, total=(6 if args.debug else len(train_loader)),
                    desc=f'ep {ep+1}/{args.epochs}', leave=True)
        for i, (x, y) in enumerate(pbar):
            opt.zero_grad()
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            loss_sum += loss.item() * y.size(0); loss_n += y.size(0)
            pbar.set_postfix(loss=f'{loss.item():.3f}')
            if args.debug and i >= 5:
                break
        sched.step()

        t1, t5, n = evaluate(model, val_loader, device, max_batches=max_b)
        print(f'Ep {ep+1}/{args.epochs}: train_loss={loss_sum/loss_n:.4f} | '
              f'val top1={t1:.2f}% top5={t5:.2f}% | {time.time()-t0:.1f}s')
        if t1 > best_top1:
            best_top1 = t1
            torch.save(model, out_dir / 'best_pruned_model.pt')  # whole model: shape changed
        torch.save(model, out_dir / 'last_pruned_model.pt')

    print('=' * 60)
    print(f'DONE. Best pruned val top1 = {best_top1:.2f}% '
          f'| params {stats["params_after"]/1e6:.2f}M ({stats["param_reduction"]:.1%} smaller)')
    print(f'FP baseline was {90.88 if args.dataset=="cifar100" else "?"}% '
          f'-> prune cost = {(90.88-best_top1):.2f}%' if args.dataset == 'cifar100' else '')
    print('=' * 60)


if __name__ == '__main__':
    main()
