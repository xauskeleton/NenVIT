"""
Smoke-test: load pretrained Swin-S from timm, run on Tiny ImageNet val (182 classes
mapped to ImageNet-1k indices), report top-1 / top-5 accuracy.

Goal: verify env, model loading, and dataloader all work end-to-end.
Expected: top-1 around 60-75% on filtered val (Swin-S full-prec on ImageNet-1k is 83.23%,
but 64x64 upsampled to 224 loses detail, so accuracy will be lower).
"""
import sys
import time
import torch
import timm

sys.path.insert(0, r'D:\xauduabo\Code+NCKH\Prune_QT_VITs\scripts')
from tiny_imagenet_loader import TinyImageNetLoaderGenerator


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct1 = correct5 = total = 0
    t0 = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            _, pred5 = logits.topk(5, dim=1)
            correct1 += (pred5[:, 0] == y).sum().item()
            correct5 += (pred5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)
            if max_batches is not None and i + 1 >= max_batches:
                break
            if (i + 1) % 20 == 0:
                print(f'  batch {i+1}: top1={100*correct1/total:.2f}%  top5={100*correct5/total:.2f}%')
    dt = time.time() - t0
    return correct1 / total, correct5 / total, total, dt


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading Swin-S pretrained from timm...')
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: swin_small_patch4_window7_224, params={n_params/1e6:.2f}M')

    print('Building Tiny ImageNet val loader...')
    gen = TinyImageNetLoaderGenerator(val_batch_size=64, num_workers=2)
    val_loader = gen.val_loader()
    print(f'Val: {len(gen.val_set)} samples, {len(val_loader)} batches')

    print('Running evaluation on full val set...')
    top1, top5, total, dt = evaluate(model, val_loader, device)
    print(f'\n=== Swin-S full-precision on Tiny ImageNet val ({total} samples, 182 classes mapped to ImageNet-1k) ===')
    print(f'Top-1: {100*top1:.2f}%')
    print(f'Top-5: {100*top5:.2f}%')
    print(f'Time:  {dt:.1f}s ({total/dt:.1f} img/s)')


if __name__ == '__main__':
    main()
