import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pathlib import Path


class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        dataset    = self.subset.dataset
        real_index = self.subset.indices[index]

        image    = Image.open(dataset.images[real_index]).convert("RGB")
        mask_rgb = Image.open(dataset.masks[real_index]).convert("RGB")

        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        image = self.transform(image)
        torch.manual_seed(seed)
        mask_tensor = self.transform(mask_rgb)

        # Brain: any non-black pixel
        brain_mask = (mask_tensor.max(dim=0)[0] > 0.05).float().unsqueeze(0)  # [1, H, W]

        return {'image': image, 'mask': brain_mask}


class UltrasoundDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)

        self.images = sorted(self.images_dir.glob('*.png'))
        self.masks  = sorted(self.masks_dir.glob('*.png'))

        if len(self.images) == 0:
            raise FileNotFoundError(f"No PNGs found in {self.images_dir.absolute()}")
        if len(self.images) != len(self.masks):
            raise ValueError(f"Image/mask count mismatch: {len(self.images)} vs {len(self.masks)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return (
            Image.open(self.images[index]).convert("RGB"),
            Image.open(self.masks[index]).convert("RGB"),
        )


def load_brain_data(images_dir="images", masks_dir="masks", batch_size=4):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = UltrasoundDataset(images_dir, masks_dir)
    train_size   = int(0.8 * len(full_dataset))
    test_size    = len(full_dataset) - train_size

    train_subset, test_subset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # same seed = same split
    )

    train_ds = SubsetWithTransform(train_subset, train_transform)
    test_ds  = SubsetWithTransform(test_subset,  test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Brain Dataset: {len(full_dataset)} total | {train_size} train | {test_size} test")
    return train_loader, test_loader