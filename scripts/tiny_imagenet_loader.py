"""
Tiny ImageNet dataloader compatible with FIMA-Q's LoaderGenerator interface.

Wraps HuggingFace `zh-plus/tiny-imagenet` (200 classes, 64x64 RGB) and remaps
labels to ImageNet-1k indices so pretrained Swin from timm works directly.
"""
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _imagenet1k_wnid_to_idx():
    """Load WordNet ID -> ImageNet-1k class index mapping from timm."""
    import timm
    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    if hasattr(info, 'label_names'):
        idx_to_wnid = info.label_names()
    else:
        idx_to_wnid = [info.index_to_label_name(i) for i in range(1000)]
    return {wnid: i for i, wnid in enumerate(idx_to_wnid)}


class TinyImageNetDataset(Dataset):
    """Wraps a HF Tiny ImageNet split, applies transform, optionally remaps labels."""

    def __init__(self, hf_split, transform, label_remap=None):
        self.hf = hf_split
        self.transform = transform
        self.label_remap = label_remap

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        item = self.hf[idx]
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        label = item['label']
        if self.label_remap is not None:
            label = self.label_remap[label]
        return img, label


class CacheDataset(Dataset):
    """Cache pre-transformed (tensor, label) pairs in RAM."""
    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, idx):
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)


class TinyImageNetLoaderGenerator:
    """
    Drop-in replacement for FIMA-Q's ImageNetLoaderGenerator using Tiny ImageNet.

    - Resizes 64 -> 224 to match Swin pretrained input
    - Maps WNID labels -> ImageNet-1k indices so pretrained head (1000 classes) works
    - Provides val_loader() and calib_loader() matching FIMA-Q's interface
    """

    def __init__(self, val_batch_size=64, num_workers=2, hf_cache_dir=None,
                 remap_to_imagenet1k=True, drop_unmappable=True):
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.remap_to_imagenet1k = remap_to_imagenet1k

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        self.train_transform = self.val_transform

        print('Loading tiny-imagenet from HuggingFace...')
        ds = load_dataset('zh-plus/tiny-imagenet', cache_dir=hf_cache_dir)
        hf_train = ds['train']
        hf_val = ds['valid']

        tiny_wnids = hf_train.features['label'].names

        if remap_to_imagenet1k:
            wnid_to_1k = _imagenet1k_wnid_to_idx()
            missing = [w for w in tiny_wnids if w not in wnid_to_1k]
            if missing and not drop_unmappable:
                raise RuntimeError(
                    f'{len(missing)} Tiny ImageNet WNIDs not in ImageNet-1k '
                    f'(first 3: {missing[:3]}). Set drop_unmappable=True to filter.')
            if missing:
                print(f'Dropping {len(missing)} classes not in ImageNet-1k: {missing}')
                keep_tiny_idx = {i for i, w in enumerate(tiny_wnids) if w in wnid_to_1k}
                hf_train = hf_train.filter(lambda ex: ex['label'] in keep_tiny_idx)
                hf_val = hf_val.filter(lambda ex: ex['label'] in keep_tiny_idx)
            self.label_remap = np.array(
                [wnid_to_1k[w] if w in wnid_to_1k else -1 for w in tiny_wnids],
                dtype=np.int64)
            kept = sum(1 for w in tiny_wnids if w in wnid_to_1k)
            print(f'Kept {kept}/{len(tiny_wnids)} classes; remapped to ImageNet-1k.')
            print(f'After filter: train={len(hf_train)}, val={len(hf_val)}')
        else:
            self.label_remap = None

        self._hf_train = hf_train
        self._hf_val = hf_val
        self.tiny_wnids = tiny_wnids
        self.num_classes = len(tiny_wnids)

        self._train_set = TinyImageNetDataset(
            self._hf_train, self.train_transform, self.label_remap)
        self._val_set = TinyImageNetDataset(
            self._hf_val, self.val_transform, self.label_remap)
        self._calib_set = None

    @property
    def train_set(self):
        return self._train_set

    @property
    def val_set(self):
        return self._val_set

    def val_loader(self):
        return DataLoader(
            self._val_set, batch_size=self.val_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=False, drop_last=False)

    def calib_loader(self, num=256, batch_size=32, seed=3, in_memory=True):
        np.random.seed(seed)
        inds = np.random.permutation(len(self._train_set))[:num]
        if in_memory:
            preloaded = [self._train_set[int(i)] for i in inds]
            self._calib_set = CacheDataset(preloaded)
        else:
            self._calib_set = torch.utils.data.Subset(self._train_set, inds.tolist())
        return DataLoader(
            self._calib_set, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)


if __name__ == '__main__':
    gen = TinyImageNetLoaderGenerator(val_batch_size=8, num_workers=0)
    print(f'Train: {len(gen.train_set)}, Val: {len(gen.val_set)}, Classes: {gen.num_classes}')

    x, y = gen.train_set[0]
    print(f'Train sample: img={tuple(x.shape)} dtype={x.dtype}, label={y} (remapped to ImageNet-1k)')

    vl = gen.val_loader()
    xb, yb = next(iter(vl))
    print(f'Val batch: imgs={tuple(xb.shape)}, labels={tuple(yb.shape)}, label range=[{yb.min().item()}, {yb.max().item()}]')

    cl = gen.calib_loader(num=64, batch_size=16)
    xc, yc = next(iter(cl))
    print(f'Calib batch: imgs={tuple(xc.shape)}, labels={tuple(yc.shape)}')
