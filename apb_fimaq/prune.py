"""
Structural pruning for Swin Transformer, guided by FIMA-Q (DPLR-FIM) importance.

Two structured granularities — both chosen because they DO NOT change a block's
input/output channel dim (stage dim), so no shape change propagates across blocks,
downsample, or norms:

  * Attention HEADS — remove whole heads from WindowAttention. qkv (3*dim,dim) loses
    each pruned head's q/k/v rows; proj (dim,dim) loses that head's input cols;
    relative_position_bias_table (n_rel, num_heads) loses that head's column;
    num_heads is decremented. Residual dim (proj.out) stays = stage dim.

  * MLP NEURONS — remove hidden neurons. fc1 (hidden,dim) loses rows; fc2 (dim,hidden)
    loses cols. fc2.out stays = stage dim.

Importance = FIMA-Q DPLR-FIM (compute_weight_dplr_fim in qat.py), aggregated:
  head h   : Σ FIM over qkv rows(q,k,v of h) + Σ FIM over proj input-cols of h
  neuron j : Σ FIM over fc1 row j           + Σ FIM over fc2 input-col j

The pruned model is a *plain* timm Swin (smaller qkv/proj/fc1/fc2, fewer heads) and
runs with timm's standard forward — no FIMA-Q wrapping needed for finetune/eval.
"""
import math
import torch
from torch import nn
import timm.models.swin_transformer as _swin
from timm.layers import Mlp


# ------------------------------------------------------------------ importance
def head_importance_from_fim(model, fim_dict):
    """Return {attn_module_name: Tensor[num_heads]} of FIM importance per head."""
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, _swin.WindowAttention):
            continue
        H = mod.num_heads
        dim = mod.qkv.in_features
        hd = dim // H
        imp = torch.zeros(H)
        f = fim_dict.get(name + '.qkv')          # (3*dim, dim)
        if f is not None:
            per_out = f.float().sum(dim=1)                  # (3*dim,)
            imp = imp + per_out.view(3, H, hd).sum(dim=(0, 2))
        f = fim_dict.get(name + '.proj')         # (dim, dim)
        if f is not None:
            per_in = f.float().sum(dim=0)                   # (dim,) input channels
            imp = imp + per_in.view(H, hd).sum(dim=1)
        out[name] = imp
    return out


def neuron_importance_from_fim(model, fim_dict):
    """Return {mlp_module_name: Tensor[hidden]} of FIM importance per MLP neuron."""
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, Mlp):
            continue
        hidden = mod.fc1.out_features
        imp = torch.zeros(hidden)
        f = fim_dict.get(name + '.fc1')          # (hidden, dim_in)
        if f is not None:
            imp = imp + f.float().sum(dim=1)                # per output neuron
        f = fim_dict.get(name + '.fc2')          # (dim_out, hidden)
        if f is not None:
            imp = imp + f.float().sum(dim=0)                # per input neuron
        out[name] = imp
    return out


# ------------------------------------------------------------------ structural cut
@torch.no_grad()
def prune_heads_in_attn(attn, keep):
    """Structurally keep only `keep` head indices in a Swin WindowAttention."""
    keep = sorted(int(k) for k in keep)
    H = attn.num_heads
    dim = attn.qkv.in_features
    hd = dim // H
    Hn = len(keep)
    dev = attn.qkv.weight.device
    dt = attn.qkv.weight.dtype

    def head_rows(offset):
        idx = []
        for h in keep:
            idx.extend(range(offset + h * hd, offset + (h + 1) * hd))
        return idx
    # grouped Q | K | V so reshape(.,3,Hn,hd) stays valid
    row_idx = torch.tensor(head_rows(0) + head_rows(dim) + head_rows(2 * dim),
                           dtype=torch.long, device=dev)
    col_idx = torch.tensor([c for h in keep for c in range(h * hd, (h + 1) * hd)],
                           dtype=torch.long, device=dev)

    new_qkv = nn.Linear(dim, 3 * Hn * hd, bias=attn.qkv.bias is not None).to(dev, dt)
    new_qkv.weight.data.copy_(attn.qkv.weight.data[row_idx])
    if attn.qkv.bias is not None:
        new_qkv.bias.data.copy_(attn.qkv.bias.data[row_idx])
    attn.qkv = new_qkv

    new_proj = nn.Linear(Hn * hd, dim, bias=attn.proj.bias is not None).to(dev, dt)
    new_proj.weight.data.copy_(attn.proj.weight.data[:, col_idx])
    if attn.proj.bias is not None:
        new_proj.bias.data.copy_(attn.proj.bias.data)
    attn.proj = new_proj

    attn.relative_position_bias_table = nn.Parameter(
        attn.relative_position_bias_table.data[:, keep].clone())
    attn.num_heads = Hn


@torch.no_grad()
def prune_neurons_in_mlp(mlp, keep):
    """Structurally keep only `keep` hidden-neuron indices in a timm Mlp."""
    assert isinstance(getattr(mlp, 'norm', nn.Identity()), nn.Identity), \
        'Mlp has a non-Identity norm on hidden dim — would need pruning too.'
    keep = torch.tensor(sorted(int(k) for k in keep), dtype=torch.long,
                        device=mlp.fc1.weight.device)
    dim_in = mlp.fc1.in_features
    dim_out = mlp.fc2.out_features
    Hn = keep.numel()
    dev = mlp.fc1.weight.device
    dt = mlp.fc1.weight.dtype

    new_fc1 = nn.Linear(dim_in, Hn, bias=mlp.fc1.bias is not None).to(dev, dt)
    new_fc1.weight.data.copy_(mlp.fc1.weight.data[keep])
    if mlp.fc1.bias is not None:
        new_fc1.bias.data.copy_(mlp.fc1.bias.data[keep])
    mlp.fc1 = new_fc1

    new_fc2 = nn.Linear(Hn, dim_out, bias=mlp.fc2.bias is not None).to(dev, dt)
    new_fc2.weight.data.copy_(mlp.fc2.weight.data[:, keep])
    if mlp.fc2.bias is not None:
        new_fc2.bias.data.copy_(mlp.fc2.bias.data)
    mlp.fc2 = new_fc2


# ------------------------------------------------------------------ selection
def _head_param_cost(attn):
    """Params owned by one head: qkv rows (3*hd*dim) + proj cols (dim*hd)."""
    dim = attn.qkv.in_features
    hd = dim // attn.num_heads
    return 4 * hd * dim


def _neuron_param_cost(mlp):
    """Params owned by one hidden neuron: fc1 row (dim_in) + fc2 col (dim_out)."""
    return mlp.fc1.in_features + mlp.fc2.out_features


def _per_layer_keep(imp_dict, ratio, floor):
    keep = {}
    for name, imp in imp_dict.items():
        n_keep = max(floor, round(len(imp) * (1 - ratio)))
        keep[name] = imp.topk(n_keep).indices.tolist()
    return keep


def _global_keep(imp_dict, cost_dict, ratio, floor_dict, metric):
    """Globally rank all units across blocks; keep top (1-ratio) by count.
    metric='per_param': score = importance / param_cost (compression-aware);
    metric='raw': score = importance.
    floor_dict[name] units (the block's own top-importance ones) are always kept,
    so global redistribution never guts a block below its floor."""
    total = sum(len(v) for v in imp_dict.values())
    n_keep_total = round(total * (1 - ratio))
    keep = {name: set() for name in imp_dict}

    forced = 0
    for name, imp in imp_dict.items():
        f = min(floor_dict.get(name, 0), len(imp))
        if f > 0:
            for j in imp.topk(f).indices.tolist():
                keep[name].add(j)
                forced += 1

    flat = []
    for name, imp in imp_dict.items():
        c = cost_dict[name] if metric == 'per_param' else 1.0
        for j, v in enumerate(imp.tolist()):
            if j not in keep[name]:
                flat.append((v / c, name, j))
    flat.sort(key=lambda t: t[0], reverse=True)
    for _, name, j in flat[:max(0, n_keep_total - forced)]:
        keep[name].add(j)
    return {name: sorted(s) for name, s in keep.items()}


# ------------------------------------------------------------------ driver
def prune_swin(model, fim_dict, head_ratio=0.25, mlp_ratio=0.25, min_heads=1,
               rank_mode='per_layer', global_metric='per_param',
               head_keep_frac=0.0, mlp_keep_frac=0.05):
    """Prune `head_ratio` of heads and `mlp_ratio` of MLP neurons, dropping the
    lowest-FIMA-Q-importance units. Returns a stats dict.

    rank_mode='per_layer': prune the same fraction within EACH block.
    rank_mode='global'    : pool all heads (resp. neurons) across blocks and prune
                            the globally lowest, so redundant blocks lose more.
                            `global_metric` = 'per_param' (importance/param-cost,
                            compression-aware) or 'raw' (importance only).

    Per-block floor (global only) — heads and MLP get SEPARATE floors:
      head floor   = max(min_heads, ceil(head_keep_frac * H))    [default: 1]
      neuron floor = max(1,        ceil(mlp_keep_frac  * hidden)) [default: 5%]
    so global can prune a block's MLP down to 5% but never gut it to 1 neuron,
    while heads are free down to a single head (min_heads).
    A floor only binds when global tries to prune that block beyond it.
    """
    assert rank_mode in ('per_layer', 'global')
    head_imp = head_importance_from_fim(model, fim_dict)
    neuron_imp = neuron_importance_from_fim(model, fim_dict)
    attns = {n: m for n, m in model.named_modules()
             if isinstance(m, _swin.WindowAttention)}
    mlps = {n: m for n, m in model.named_modules() if isinstance(m, Mlp)}
    p0 = sum(p.numel() for p in model.parameters())

    if rank_mode == 'per_layer':
        head_keep = _per_layer_keep(head_imp, head_ratio, min_heads)
        neur_keep = _per_layer_keep(neuron_imp, mlp_ratio, 1)
    else:
        head_floor = {n: max(min_heads, math.ceil(head_keep_frac * m.num_heads))
                      for n, m in attns.items()}
        neur_floor = {n: max(1, math.ceil(mlp_keep_frac * m.fc1.out_features))
                      for n, m in mlps.items()}
        head_keep = _global_keep(head_imp, {n: _head_param_cost(m) for n, m in attns.items()},
                                 head_ratio, head_floor, global_metric)
        neur_keep = _global_keep(neuron_imp, {n: _neuron_param_cost(m) for n, m in mlps.items()},
                                 mlp_ratio, neur_floor, global_metric)

    n_heads_before = sum(m.num_heads for m in attns.values())
    n_neur_before = sum(m.fc1.out_features for m in mlps.values())
    n_heads_after = sum(len(v) for v in head_keep.values())
    n_neur_after = sum(len(v) for v in neur_keep.values())

    for name, mod in attns.items():
        prune_heads_in_attn(mod, head_keep[name])
    for name, mod in mlps.items():
        prune_neurons_in_mlp(mod, neur_keep[name])

    p1 = sum(p.numel() for p in model.parameters())
    return {
        'params_before': p0, 'params_after': p1,
        'param_reduction': 1 - p1 / p0,
        'heads_before': n_heads_before, 'heads_after': n_heads_after,
        'mlp_neurons_before': n_neur_before, 'mlp_neurons_after': n_neur_after,
        'rank_mode': rank_mode,
    }


# ------------------------------------------------------------------ self-test
if __name__ == '__main__':
    import timm
    from qat import get_target_layers

    def heads_per_stage(model):
        from collections import OrderedDict
        d = OrderedDict()
        for n, mod in model.named_modules():
            if isinstance(mod, _swin.WindowAttention):
                stage = n.split('.')[1]            # layers.<stage>.blocks...
                d[stage] = d.get(stage, 0) + mod.num_heads
        return dict(d)

    x = torch.randn(2, 3, 224, 224)
    for mode in ('per_layer', 'global'):
        torch.manual_seed(0)
        m = timm.create_model('swin_small_patch4_window7_224', num_classes=100).eval()
        fake_fim = {n: torch.rand_like(mod.weight)
                    for n, mod in get_target_layers(m, scope='skip').items()}
        stats = prune_swin(m, fake_fim, head_ratio=0.25, mlp_ratio=0.25,
                           rank_mode=mode, global_metric='per_param')
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 100) and torch.isfinite(y).all()
        print(f'[{mode:9s}] forward OK | heads {stats["heads_before"]}->'
              f'{stats["heads_after"]} | mlp {stats["mlp_neurons_before"]}->'
              f'{stats["mlp_neurons_after"]} | params '
              f'{stats["params_after"]/1e6:.2f}M ({stats["param_reduction"]:.1%} smaller) '
              f'| heads/stage {heads_per_stage(m)}')
