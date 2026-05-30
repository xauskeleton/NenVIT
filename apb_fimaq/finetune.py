"""
Plain FP fine-tuning of Swin-S — no APB / FIM / DPLR.

Produces a clean full-precision baseline (best.pth = raw model.state_dict()).
Shares data loaders, eval and log-tee with qat.py so numbers are directly
comparable. (qat.py currently starts QAT from timm-pretrained weights, not from
this checkpoint — wiring an --init-from flag would be a separate change.)

USAGE
=====
    python finetune.py --dataset cifar100 --epochs 20 --out-dir ckpt/ft_cifar100
    python finetune.py --dataset tiny --epochs 10
    python finetune.py --dataset cifar10 --debug         # quick smoke
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

# Data loaders imported lazily in build_data() per --dataset (the tiny loader
# pulls HuggingFace `datasets`, which a CIFAR-only run should not require).
from qat import evaluate, _Tee  # noqa: E402  (reuse identical eval + log tee)

MODEL = 'swin_small_patch4_window7_224'


def build_data(args):
    if args.dataset == 'tiny':
        from tiny_imagenet_loader import TinyImageNetLoaderGenerator
        g = TinyImageNetLoaderGenerator(val_batch_size=args.batch_size,
                                        num_workers=args.num_workers)
        num_classes = 1000
    elif args.dataset == 'imagenet1k':
        if not args.data_dir:
            raise ValueError('--data-dir is required when --dataset=imagenet1k')
        from imagenet1k_loader import ImageNet1kLoaderGenerator
        g = ImageNet1kLoaderGenerator(data_dir=args.data_dir,
                                      val_batch_size=args.batch_size,
                                      num_workers=args.num_workers)
        num_classes = 1000
    elif args.dataset in ('cifar10', 'cifar100'):
        from cifar_loader import CifarLoaderGenerator
        g = CifarLoaderGenerator(which=args.dataset, data_dir=args.data_dir,
                                 val_batch_size=args.batch_size,
                                 num_workers=args.num_workers)
        num_classes = g.num_classes
    else:
        raise ValueError(f'unknown dataset {args.dataset!r}')

    train_loader = torch.utils.data.DataLoader(
        g.train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return train_loader, g.val_loader(), num_classes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=['tiny', 'imagenet1k', 'cifar10', 'cifar100'],
                   default='cifar100',
                   help='Dataset (default cifar100). cifar*/torchvision auto-download; '
                        'tiny/imagenet1k as in qat.py.')
    p.add_argument('--data-dir', type=str, default='',
                   help='Root dir. Required for imagenet1k; download dir for cifar*.')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5,
                   help='Early stopping: stop if val top1 does not improve by '
                        '>--min-delta for this many epochs (default 5; 0 = disabled). '
                        'best.pth is always kept.')
    p.add_argument('--min-delta', type=float, default=0.0,
                   help='Minimum val top1 gain (percentage points) to count as an '
                        'improvement for early stopping (default 0.0).')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--freeze-backbone', action='store_true',
                   help='Train only the classification head (fast linear-probe baseline).')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                   help='CUDA mixed-precision (fp16). On by default; disable with --no-amp.')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--debug', action='store_true',
                   help='Quick mode: 2 epochs, 6 train batches/epoch, tiny val eval.')
    p.add_argument('--out-dir', type=str,
                   default=str(PROJECT_ROOT / 'checkpoints' / 'finetune'))
    p.add_argument('--log-file', type=str, default='',
                   help='Log path. Default <out-dir>/train.log; "none" to disable.')
    p.add_argument('--resume', type=str, default='',
                   help='Path to last.pth to resume (model+opt+sched+scaler+epoch).')
    args = p.parse_args()

    if args.debug:
        # cap batch (not force) so low-VRAM GPUs can smoke; honors smaller --batch-size
        args.epochs = 2; args.batch_size = min(args.batch_size, 16); args.num_workers = 0

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_path = args.log_file or str(out_dir / 'train.log')
    if log_path.lower() != 'none':
        log_fp = open(log_path, 'a', buffering=1, encoding='utf-8')
        sys.stdout = _Tee(sys.stdout, log_fp)
        print(f'\n[{datetime.now():%Y-%m-%d %H:%M:%S}] ===== finetune start '
              f'(log → {log_path}) =====')

    print('=' * 60)
    print(f'FP fine-tune Swin-S | device={device}')
    print(f'Args: {vars(args)}')
    print('=' * 60)

    train_loader, val_loader, num_classes = build_data(args)

    print(f'Loading Swin-S pretrained (num_classes={num_classes}) ...')
    model = timm.create_model(MODEL, pretrained=True, num_classes=num_classes)
    model.to(device)

    if args.freeze_backbone:
        head = model.get_classifier()
        head_params = set(head.parameters())
        for prm in model.parameters():
            prm.requires_grad = prm in head_params
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Backbone frozen — training head only ({n_train/1e6:.2f}M params).')

    max_b = 10 if args.debug else None
    t1, t5, n = evaluate(model, val_loader, device, max_batches=max_b)
    print(f'[Pretrained baseline] top1={t1:.2f}% top5={t5:.2f}% ({n} samples) '
          f'— head is random for cifar*, expect ~chance.')

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    use_amp = bool(args.amp) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_top1 = 0.0
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        if use_amp and ck.get('scaler') is not None:
            scaler.load_state_dict(ck['scaler'])
        start_epoch = ck['epoch'] + 1
        best_top1 = ck.get('best_top1', 0.0)
        print(f'>>> Resumed from {args.resume}: epoch {start_epoch}/{args.epochs}, '
              f'best_top1={best_top1:.2f}%')
        if start_epoch >= args.epochs:
            print(f'>>> Already finished ({start_epoch} >= {args.epochs}).')
            return

    print('=' * 60)
    es = f'patience={args.patience}' if args.patience > 0 else 'off'
    print(f'Fine-tuning: {args.epochs} epochs (start={start_epoch}), lr={args.lr}, '
          f'wd={args.weight_decay}, ls={args.label_smoothing}, amp={use_amp}, '
          f'early-stop={es}')
    print('=' * 60)

    epochs_no_improve = 0
    for ep in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time(); loss_sum = 0; loss_n = 0
        n_batches = 6 if args.debug else len(train_loader)
        pbar = tqdm(train_loader, total=n_batches, desc=f'ep {ep+1}/{args.epochs}',
                    leave=True)
        for i, (x, y) in enumerate(pbar):
            opt.zero_grad()
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * y.size(0); loss_n += y.size(0)
            pbar.set_postfix(loss=f'{loss.item():.3f}')
            if args.debug and i >= 5:
                break
        sched.step()

        t1, t5, n = evaluate(model, val_loader, device, max_batches=max_b)
        dt = time.time() - t0
        print(f'Ep {ep+1}/{args.epochs}: train_loss={loss_sum/loss_n:.4f} | '
              f'val top1={t1:.2f}% top5={t5:.2f}% | {dt:.1f}s')

        improved = t1 > best_top1 + args.min_delta
        if t1 > best_top1:
            best_top1 = t1
            torch.save(model.state_dict(), out_dir / 'best.pth')
        torch.save({
            'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'scaler': scaler.state_dict() if use_amp else None,
            'best_top1': best_top1,
        }, out_dir / 'last.pth')

        if args.patience > 0:
            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f'  >> no improvement ({epochs_no_improve}/{args.patience}) '
                      f'| best={best_top1:.2f}%')
                if epochs_no_improve >= args.patience:
                    print(f'  >> Early stopping at epoch {ep+1} '
                          f'(no val top1 gain >{args.min_delta} for {args.patience} epochs).')
                    break

    print('=' * 60)
    print(f'DONE. Best val top1 = {best_top1:.2f}%')
    print('=' * 60)


if __name__ == '__main__':
    main()
