"""
CIFAR-10 / CIFAR-100 dataloader matching TinyImageNetLoaderGenerator interface.

Swin-S is fixed at 224x224 (patch4/window7), so CIFAR's native 32x32 images are
upsampled to 224 (bicubic) and normalized with ImageNet stats — this lets the
ImageNet-pretrained backbone transfer. The classification head is created fresh
for 10/100 classes (see --dataset handling in qat.py), so it must be trained.

Downloads via torchvision (no HuggingFace `datasets` dependency).
"""
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100


class CifarLoaderGenerator:
    """Drop-in replacement for TinyImageNetLoaderGenerator using CIFAR-10/100."""
    def __init__(self, which='cifar100', data_dir='', val_batch_size=64,
                 num_workers=2, train_augment=True):
        which = which.lower()
        if which == 'cifar100':
            ds_cls, self.num_classes = CIFAR100, 100
        elif which == 'cifar10':
            ds_cls, self.num_classes = CIFAR10, 10
        else:
            raise ValueError(f'which must be cifar10|cifar100, got {which!r}')

        root = data_dir or './data'
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        bicubic = transforms.InterpolationMode.BICUBIC
        self.val_transform = transforms.Compose([
            transforms.Resize(224, interpolation=bicubic),
            transforms.ToTensor(),
            normalize,
        ])
        if train_augment:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224, interpolation=bicubic),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.train_transform = self.val_transform

        print(f'Loading {which} (torchvision) into {root} ...')
        self._train_set = ds_cls(root=root, train=True, download=True,
                                 transform=self.train_transform)
        self._val_set = ds_cls(root=root, train=False, download=True,
                               transform=self.val_transform)
        print(f'{which}: train={len(self._train_set)}, val={len(self._val_set)}, '
              f'classes={self.num_classes}')

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