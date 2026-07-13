"""
Backfill wandb tu cac file log .md trong runs/.

Moi file runs/*.md -> mot wandb run:
  - parse bang "Per-epoch" (nhieu dinh dang khac nhau) -> log loss/ce/top1/top5/epoch_time theo epoch
  - parse block ```Config``` (key = value) + dong Command (cac --flag) -> wandb.config
  - parse bang Summary (FP baseline / best) -> summary metrics

KHONG dung toi code train. Chi doc file .md va day len wandb.

Cach chay (offline, khong can tai khoan):
    WANDB_MODE=offline python scripts/backfill_wandb.py

Cach chay (online, da `wandb login`):
    python scripts/backfill_wandb.py --project prune-qt-vits

Sau khi chay offline, day len web bang:
    wandb sync wandb/offline-run-*
"""
import argparse
import glob
import os
import re
import sys

import wandb

# --------------------------------------------------------------------------
# Regex cho 1 dong epoch. Bat cac dinh dang gap trong runs/*.md:
#   "Ep  1: loss=4.4745 (ce=3.0365 + lambda.dplr=0.0005) | top1=47.72% top5=80.01% | 1442.5s"
#   "Ep  2: loss=3.0463 (ce=1.7317)                       | top1=58.09% top5=86.77% | 1456.5s"
#   "Ep  1: train_loss=1.2003 | val top1=87.08% top5=98.40% | 440.3s"
#   "Ep  1/20: loss=2.4783 (ce=1.0124 + lambda.dplr=0.0005) | val top1=81.96% top5=95.57% | 1340.3s"
# --------------------------------------------------------------------------
EPOCH_RE = re.compile(
    r"""^\s*Ep\s+(?P<epoch>\d+)(?:\s*/\s*\d+)?\s*:\s*"""      # Ep 12  hoac Ep 12/20
    r"""(?:train_)?loss=(?P<loss>[\d.]+)"""                    # loss= hoac train_loss=
    r"""(?:\s*\(ce=(?P<ce>[\d.]+)"""                           # (ce=...
    r"""(?:\s*\+\s*[^)=]*dplr=(?P<dplr>[\d.eE+-]+))?\s*\))?"""  # + lambda.dplr=...)
    r""".*?top1=(?P<top1>[\d.]+)%"""                           # top1=..%
    r"""(?:\s*top5=(?P<top5>[\d.]+)%)?"""                       # top5=..% (tuy chon)
    r"""(?:.*?(?P<sec>[\d.]+)s)?""",                            # 1442.5s (tuy chon)
    re.VERBOSE,
)

CONFIG_LINE_RE = re.compile(r"^\s*([a-zA-Z_][\w]*)\s*=\s*(.+?)\s*$")
FLAG_RE = re.compile(r"--([a-zA-Z0-9][\w-]*)(?:[= ]+([^\s\\]+))?")
SUMMARY_ROW_RE = re.compile(r"\|(.+?)\|\s*([\d.]+)%\s*\|\s*([\d.]+)%")


def _num(s):
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except (ValueError, AttributeError):
        return s


def parse_epochs(text):
    """Tra ve list dict metric theo epoch."""
    rows = []
    for line in text.splitlines():
        m = EPOCH_RE.match(line)
        if not m:
            continue
        d = m.groupdict()
        row = {
            "epoch": int(d["epoch"]),
            "loss": float(d["loss"]),
            "top1": float(d["top1"]),
        }
        if d.get("top5") is not None:
            row["top5"] = float(d["top5"])
        if d.get("ce") is not None:
            row["ce"] = float(d["ce"])
        if d.get("dplr") is not None:
            row["dplr_raw"] = float(d["dplr"])
        if d.get("sec") is not None:
            row["epoch_time_s"] = float(d["sec"])
        rows.append(row)
    return rows


def parse_config(text):
    """Ghep config tu block ```Config``` (key = value) + cac --flag trong Command."""
    cfg = {}

    # 1) Cac fenced code block: lay dong "key = value"
    for block in re.findall(r"```(.*?)```", text, re.DOTALL):
        # bo qua block command bash (co dau \ noi dong / bat dau bang 'python')
        for line in block.splitlines():
            m = CONFIG_LINE_RE.match(line)
            if m:
                key, val = m.group(1), m.group(2)
                # cat phan comment "# ..." o cuoi gia tri
                val = val.split("#")[0].strip()
                # bo cac dong nhieu (vd '============')
                if key and val and not set(val) <= {"="}:
                    cfg[key] = _num(val)

    # 2) Cac --flag trong file (Command). Ghi de neu trung, uu tien flag ro rang.
    for fm in FLAG_RE.finditer(text):
        flag, val = fm.group(1), fm.group(2)
        key = flag.replace("-", "_")
        if val is None:
            cfg.setdefault(key, True)  # store_true flag
        else:
            cfg[key] = _num(val)

    return cfg


def parse_summary(text):
    """Lay top1/top5 tu bang Summary. Tra ve dict {stage_label: (top1, top5)}."""
    out = {}
    for m in SUMMARY_ROW_RE.finditer(text):
        label = m.group(1).strip().strip("*").strip()
        label = re.sub(r"\s+", "_", label.lower())
        label = re.sub(r"[^\w]", "", label)
        if label:
            out[label] = (float(m.group(2)), float(m.group(3)))
    return out


def process_file(path, project, entity, dry_run):
    with open(path, encoding="utf-8") as f:
        text = f.read()

    name = os.path.splitext(os.path.basename(path))[0]
    epochs = parse_epochs(text)
    cfg = parse_config(text)
    summary = parse_summary(text)

    # tieu de (dong # dau tien) lam ghi chu
    title_m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    notes = title_m.group(1).strip() if title_m else ""

    print(f"\n=== {name}")
    print(f"    epochs parsed : {len(epochs)}")
    print(f"    config keys   : {len(cfg)}")
    print(f"    summary rows  : {len(summary)}")
    if not epochs:
        print("    [!] khong parse duoc epoch nao -> BO QUA (van co the them config-only run neu muon)")

    if dry_run:
        if epochs:
            print(f"    first epoch   : {epochs[0]}")
            print(f"    last  epoch   : {epochs[-1]}")
        return len(epochs)

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        notes=notes,
        config=cfg,
        reinit=True,
        tags=["backfill"],
    )
    for row in epochs:
        run.log(row, step=row["epoch"])

    # summary: best top1/top5 tu per-epoch + cac stage tu bang Summary
    if epochs:
        best = max(epochs, key=lambda r: r["top1"])
        run.summary["best_top1"] = best["top1"]
        if "top5" in best:
            run.summary["best_top5"] = best["top5"]
        run.summary["best_epoch"] = best["epoch"]
    for label, (t1, t5) in summary.items():
        run.summary[f"summary_{label}_top1"] = t1
        run.summary[f"summary_{label}_top5"] = t5

    run.finish()
    return len(epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--project", default="prune-qt-vits")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--glob", default="*.md")
    ap.add_argument("--dry-run", action="store_true",
                    help="Chi parse va in ra, KHONG goi wandb")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.runs_dir, args.glob)))
    # bo README.md (khong phai run)
    paths = [p for p in paths if os.path.basename(p).lower() != "readme.md"]

    if not paths:
        print(f"Khong tim thay file nao khop {args.runs_dir}/{args.glob}")
        sys.exit(1)

    print(f"Tim thay {len(paths)} file. dry_run={args.dry_run} "
          f"mode={os.environ.get('WANDB_MODE', 'online')}")

    total_epochs = 0
    n_with_epochs = 0
    for p in paths:
        n = process_file(p, args.project, args.entity, args.dry_run)
        total_epochs += n
        n_with_epochs += (n > 0)

    print(f"\n--- XONG. {n_with_epochs}/{len(paths)} file co per-epoch data, "
          f"tong {total_epochs} epoch-rows.")


if __name__ == "__main__":
    main()
