"""
ImageNet-1k dataloader matching TinyImageNetLoaderGenerator interface.

Reads ImageFolder-style data:
    <data_dir>/train/<wnid>/*.JPEG
    <data_dir>/val/<wnid>/*.JPEG

Labels = ImageFolder's alphabetical class index. To make Swin pretrained heads
work directly, we remap to ImageNet-1k's standard wnid->index ordering (from timm).
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def _imagenet1k_wnid_to_idx():
    """Standard ImageNet-1k wnid -> index mapping from timm."""
    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    if hasattr(info, 'label_names'):
        idx_to_wnid = info.label_names()
    else:
        idx_to_wnid = [info.index_to_label_name(i) for i in range(1000)]
    return {wnid: i for i, wnid in enumerate(idx_to_wnid)}


class _RemappedImageFolder(Dataset):
    """Wrap ImageFolder, remap its alphabetical labels to standard ImageNet-1k indices."""
    def __init__(self, root, transform):
        self.base = ImageFolder(root, transform=transform)
        wnid_to_1k = _imagenet1k_wnid_to_idx()
        # ImageFolder.classes is sorted alphabetically; build a permutation
        self.remap = np.array(
            [wnid_to_1k.get(c, -1) for c in self.base.classes], dtype=np.int64)
        missing = [c for c, r in zip(self.base.classes, self.remap) if r < 0]
        if missing:
            raise RuntimeError(
                f'{len(missing)} ImageFolder classes not in standard ImageNet-1k '
                f'wnid list (first 3: {missing[:3]}). Check data_dir structure.')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, lbl = self.base[idx]
        return img, int(self.remap[lbl])


class ImageNet1kLoaderGenerator:
    """
    Drop-in replacement for TinyImageNetLoaderGenerator using full ImageNet-1k.

    Expects ImageFolder layout:
        <data_dir>/train/<wnid>/*.JPEG
        <data_dir>/val/<wnid>/*.JPEG
    """
    def __init__(self, data_dir, val_batch_size=64, num_workers=2,
                 train_augment=True):
        data_dir = Path(data_dir)
        train_dir = data_dir / 'train'
        val_dir = data_dir / 'val'
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(
                f'Expected {train_dir} and {val_dir} (ImageFolder layout). '
                f'On Kaggle, point --data-dir at the folder containing train/ and val/.')

        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        if train_augment:
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.train_transform = self.val_transform

        print(f'Loading ImageNet-1k from {data_dir}...')
        self._train_set = _RemappedImageFolder(str(train_dir), self.train_transform)
        self._val_set = _RemappedImageFolder(str(val_dir), self.val_transform)
        self.num_classes = 1000
        print(f'ImageNet-1k: train={len(self._train_set)}, val={len(self._val_set)}')

    @property
    def train_set(self):
        return self._train_set

    @property
    def val_set(self):
        return self._val_set

    def val_loader(self):
        return DataLoader(
            self._val_set, batch_size=self.val_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)