
# NenVIT — Nén Vision Transformer: Structural Pruning + APB Quantization

Nén **Swin / DeiT / ViT** bằng hai hướng, cùng dựa trên Fisher Information:

1. **Structural pruning** — bỏ attention head + MLP neuron theo empirical Fisher (`prune_cifar.py`)
2. **APB quantization + QAT** — chia weight thành binary (`α·sign(w)`) và full-precision rồi
   fine-tune, kèm distillation loss DPLR-FIM (`qat.py`)

Ghép được thành chuỗi **prune → finetune → quantize**.

Paper nền: [APB](https://arxiv.org/abs/2306.08960) · [FIMA-Q](https://arxiv.org/abs/2506.11543) — PDF trong `pdf/`.

---

## Cài đặt

Cần GPU NVIDIA. Đã test trên Python 3.10, torch 2.11 + CUDA 12.8, timm 1.0.27.

```bash
conda create -n fimaq_apb python=3.10 -y
conda activate fimaq_apb

# torch theo dung CUDA cua may -- xem https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

```bash
python -c "import torch, timm; print(torch.__version__, torch.cuda.is_available(), timm.__version__)"
```

**Dữ liệu:** CIFAR-100 tự tải về `./data` ở lần chạy đầu (~170 MB), không phải chuẩn bị gì.

---

## Chạy

Chạy từ thư mục gốc của repo.

```bash
# 0) Smoke test (~1 phut)
python apb_fimaq/qat.py --debug --dataset cifar100

# 1) FP baseline -- bat buoc, moi kien truc mot baseline rieng
python apb_fimaq/finetune.py --dataset cifar100 --epochs 20 --out-dir ckpt

# 2) Structural pruning (bo ~50% head + MLP neuron)
python apb_fimaq/prune_cifar.py --dataset cifar100 --baseline-ckpt ckpt/best.pth \
    --importance fisher --rank-mode global --head-ratio 0.5 --mlp-ratio 0.5 \
    --epochs 20 --out-dir ckpt/pruned

# 3) APB quantization + QAT tren model da prune
python apb_fimaq/qat.py --dataset cifar100 --init-model ckpt/pruned/best_pruned_model.pt \
    --partition magnitude --binary-ratio 0.99 --use-dplr-loss --dplr-lambda 0.1 \
    --epochs 30 --out-dir ckpt/prune_quant
```

Cả ba bước bằng một lệnh (tự bỏ qua bước đã xong):
```bash
MODEL=swin_small_patch4_window7_224 PY=python bash scripts/run_e2e.sh
```

Đổi kiến trúc bằng `--model` (`deit_small_patch16_224`, `vit_small_patch16_224`, …).
Mỗi kiến trúc cần baseline riêng ở bước 1, không dùng chung được.

### Cờ hay dùng

| cờ | mặc định | |
|---|---|---|
| `--binary-ratio` | 0.75 | tỉ lệ weight bị binarize; 0.95–0.99 là vùng hay dùng |
| `--partition` | `magnitude` | chọn weight giữ FP: `magnitude` hoặc `fisher` |
| `--use-dplr-loss` | tắt | distillation loss per-block từ teacher FP (DPLR-FIM) |
| `--dplr-lambda` | 0.1 | trọng số DPLR; loss đã chuẩn hóa về O(1) nên 0.1 là điểm cân |
| `--act-bits` | 0 | lượng tử hóa activation (0 = tắt; 8/4/2/1) |
| `--epochs` `--batch-size` `--lr` | 10 / 64 / 1e-4 | |
| `--no-amp` | | tắt mixed precision |
| `--out-dir` | | **luôn đặt riêng cho mỗi run** — mặc định sẽ ghi đè |

`--help` để xem hết — còn `--dplr-temp`, `--dplr-rank`, `--dplr-calib-size`, và
`--dplr-legacy-loss` (khôi phục DPLR loss cũ, cần `--dplr-lambda 3000`).

---

## APBLinear

```python
mask=True  →  α · sign(latent_weight)     # 1 bit
mask=False →  latent_weight               # 32 bit
```

`α` học được (custom STE giữ gradient cho cả `latent_weight` lẫn `α`), đóng băng ở nửa chừng
training. Mask cố định suốt run. Weight decay không áp lên `α`. Bọc toàn bộ 100 `nn.Linear`
kể cả `head.fc` và `downsample`; chỉ bỏ `patch_embed` (Conv2d).

`--export-packed` lưu thêm `best_packed.pt` theo định dạng Eq 10 của paper APB.

---

## Cấu trúc

```
apb_fimaq/
  qat.py            APB quantization + QAT (APBLinear, FIM, DPLR loss, packed export)
  prune.py          thu vien cat head / MLP neuron
  prune_cifar.py    driver: prune + finetune
  finetune.py       tao FP baseline
scripts/
  cifar_loader.py   CIFAR-10/100 (32 -> 224 bicubic)
  run_e2e.sh        chay ca 3 buoc
runs/               log tung run + bang ket qua tong hop
FIMA-Q/             repo goc, chi import -- khong sua
pdf/                cac paper nen
```

Checkpoint, `data/`, `wandb/` nằm trong `.gitignore`.

---

## Đọc thêm

| file | nội dung |
|---|---|
| `runs/README.md` | bảng kết quả tổng hợp — **nguồn số chính thức** |
| `ABLATIONS.md` | thiết kế + kết quả các ablation |

Repo: https://github.com/xauskeleton/NenVIT
