from datasets import load_dataset

print('Downloading tiny-imagenet from HuggingFace...')
ds = load_dataset('zh-plus/tiny-imagenet')
print(f'Splits: {list(ds.keys())}')
for split in ds:
    print(f'  {split}: {len(ds[split])} samples')
print(f'Features: {ds["train"].features}')
print(f'Num classes: {ds["train"].features["label"].num_classes}')

sample = ds['train'][0]
print(f'Sample image type: {type(sample["image"])}, size: {sample["image"].size}, mode: {sample["image"].mode}')
print(f'Sample label: {sample["label"]}')
