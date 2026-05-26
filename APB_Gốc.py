import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms, datasets
import numpy as np
import os
import time

# --- CONFIG ---
CONFIG = {
    "epochs": 50, "batch_size": 128, "lr": 1e-3, "weight_decay": 5e-4, "momentum": 0.9,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "resnet18_apb_original.pth"
}


# --- LAYER: APB GỐC (NO ZERO) ---
class APBLayerOriginal(nn.Module):
    def __init__(self, layer_to_wrap: nn.Module):
        super().__init__()
        self.wrapped_layer = layer_to_wrap
        self.latent_weight = nn.Parameter(layer_to_wrap.weight.data.clone())
        self.bias = layer_to_wrap.bias
        if hasattr(self.wrapped_layer, 'weight'): del self.wrapped_layer.weight

        with torch.no_grad():
            w = self.latent_weight.data
            self.alpha = nn.Parameter(w.abs().mean())
            self.delta = nn.Parameter(3.0 * w.std().clamp(min=1e-5))

    def forward(self, x):
        t = self.alpha.abs() + self.delta.clamp(min=1e-8)
        w_abs = self.latent_weight.abs()
        sign = torch.sign(self.latent_weight)

        # Chỉ có FP và Binary (+-alpha)
        fp_mask = w_abs > t
        binary_part = sign * self.alpha.abs()
        eff_weight = torch.where(fp_mask, self.latent_weight, binary_part)

        # STE
        out = self.latent_weight + (eff_weight - self.latent_weight).detach()

        if isinstance(self.wrapped_layer, nn.Linear): return F.linear(x, out, self.bias)
        return F.conv2d(x, out, self.bias, self.wrapped_layer.stride, self.wrapped_layer.padding,
                        self.wrapped_layer.dilation, self.wrapped_layer.groups)


def apply_apb(model):
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if name == "conv1" or name == "fc" or "downsample" in name:
                continue
            parent = model
            for part in name.split('.')[:-1]: parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], APBLayerOriginal(m))
    return model.to(CONFIG['device'])


# --- SAVE FUNCTION ---
def save_best_model(model, path):
    state = model.state_dict().copy()
    total_params = 0
    count_fp = 0

    for name, m in model.named_modules():
        if isinstance(m, APBLayerOriginal):
            flat_w = m.latent_weight.data.view(-1)
            t = m.alpha.abs() + m.delta.clamp(min=1e-8)

            # Thống kê
            num_fp = (flat_w.abs() > t).sum().item()
            count_fp += num_fp
            total_params += flat_w.numel()

            # Lưu trữ
            sign_bits = (flat_w > 0).to(torch.uint8)
            packed_sign = torch.from_numpy(np.packbits(sign_bits.cpu().numpy()))

            mask_fp = flat_w.abs() > t
            idx_fp = torch.nonzero(mask_fp, as_tuple=False).view(-1)
            val_fp = flat_w[mask_fp]
            idx_fp = idx_fp.to(torch.int16) if flat_w.numel() < 65535 else idx_fp.to(torch.int32)

            state[f"{name}.packed_sign"] = packed_sign
            state[f"{name}.fp_idx"] = idx_fp
            state[f"{name}.fp_val"] = val_fp
            del state[f"{name}.latent_weight"], state[f"{name}.delta"]

    torch.save(state, path)
    size_mb = os.path.getsize(path) / (1024 ** 2)

    # In tỷ lệ
    fp_ratio = 100 * count_fp / total_params
    bin_ratio = 100 - fp_ratio
    print(f"--> New Best! Size: {size_mb:.2f} MB | FP: {fp_ratio:.2f}% | Binary: {bin_ratio:.2f}%")


# --- MAIN ---
def main():
    print(f"Running APB Original on {CONFIG['device']}")
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                            transforms.Normalize(*stats)])

    # FIX: Dùng keyword argument download=True
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=t)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*stats)]))

    train_l = torch.utils.data.DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    test_l = torch.utils.data.DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    model = apply_apb(models.resnet18(weights='DEFAULT'))
    model.fc = nn.Linear(model.fc.in_features, 10);
    model.to(CONFIG['device'])

    opt = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'],
                    weight_decay=CONFIG['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG['epochs'])
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(CONFIG['epochs']):
        if ep == CONFIG['epochs'] // 2:
            print("Freeze Params")
            for m in model.modules():
                if hasattr(m, 'alpha'): m.alpha.requires_grad = m.delta.requires_grad = False

        model.train()
        for x, y in train_l:
            opt.zero_grad();
            loss = crit(model(x.to(CONFIG['device'])), y.to(CONFIG['device']));
            loss.backward();
            opt.step()
        sched.step()

        model.eval()
        corr = 0;
        tot = 0
        with torch.no_grad():
            for x, y in test_l:
                out = model(x.to(CONFIG['device']))
                _, p = torch.max(out, 1);
                tot += y.size(0);
                corr += (p == y.to(CONFIG['device'])).sum().item()
        acc = 100 * corr / tot
        print(f"Ep {ep + 1}: Acc={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            save_best_model(model, CONFIG['save_path'])

        # In stats nhanh mỗi epoch để theo dõi
        if (ep + 1) % 1 == 0:
            m = list(model.modules())[2]  # Lấy layer đầu tiên (conv2 của resnet)
            if isinstance(m, APBLayerOriginal):
                t = m.alpha.abs() + m.delta.clamp(min=1e-8)
                fp_pct = (m.latent_weight.abs() > t).float().mean() * 100
                print(f"   [Stats Layer 1] FP: {fp_pct:.2f}% | Binary: {100 - fp_pct:.2f}%")

    print("=" * 60)
    print(f"TRAINING COMPLETE.")
    print(f"Best Accuracy: {best_acc:.2f}%")

    # Tính kích thước file
    file_size = os.path.getsize(CONFIG['save_path']) / (1024 * 1024)
    print(f"Original APB Model Size: {file_size:.2f} MB")
    print("=" * 60)

if __name__ == "__main__": main()