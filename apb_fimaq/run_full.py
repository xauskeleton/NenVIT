"""
One-shot launcher — chạy qat.py với TẤT CẢ flag đầy đủ.

Chỉ cần:
    python run_full.py

(Không cần truyền flag gì, mọi thứ đã set sẵn bên trong.)

Để chỉnh, sửa block CONFIG dưới đây hoặc dùng qat.py trực tiếp.
"""
import sys
from pathlib import Path

# ============================================================
# CONFIG — chỉnh ở đây
# ============================================================
CONFIG = {
    # APB
    'apb_scope':          'skip',  # 'skip' = 96 layers (safe) | 'full' = 100 layers (incl head + downsample)
    'binary_ratio':       0.75,    # % weights binarize
    # Training
    'epochs':             20,      # số epoch
    'lr':                 1e-4,    # learning rate AdamW
    'batch_size':         64,      # batch size train/val
    # FIM extraction
    'fim_batches':        5,       # = rank k (paper default)
    'importance':         'dplr',  # 'dplr' | 'fisher'
    'fim_p1':             1.0,     # weight rank-k (L2)
    'fim_p2':             1.0,     # weight diag (L1)
    # Dynamic mask refresh — DISABLED for stability
    # Lý do: flip binary↔FP gây spike output (mất ổn định training).
    # Mask cố định từ FIM init, chỉ sign (+α/-α) thay đổi qua training.
    'recompute_fim_every': 0,
    # DPLR distillation loss — guide gradient để quyết định sign (+α/-α) per weight
    'use_dplr_loss':      True,
    'dplr_lambda':        3000,    # sweet spot từ ablation debug
    # System
    'num_workers':        2,
    'seed':               3407,
    'out_dir':            None,    # None → default ./checkpoints/qat
}

# ============================================================
def main():
    # Build sys.argv from CONFIG, then defer to qat.main()
    args = ['qat.py']
    for k, v in CONFIG.items():
        flag = '--' + k.replace('_', '-')
        if isinstance(v, bool):
            if v:
                args.append(flag)
        elif v is not None:
            args += [flag, str(v)]

    print('=' * 60)
    print('RUN_FULL — qat.py với cấu hình:')
    for k, v in CONFIG.items():
        print(f'  {k:25s} = {v}')
    print('=' * 60)
    print('Equivalent CLI:')
    print('  python qat.py ' + ' '.join(args[1:]))
    print('=' * 60)

    sys.argv = args
    sys.path.insert(0, str(Path(__file__).parent))
    from qat import main as qat_main
    qat_main()


if __name__ == '__main__':
    main()
