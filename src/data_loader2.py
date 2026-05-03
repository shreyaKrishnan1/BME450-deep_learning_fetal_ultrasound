import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pathlib import Path


class UltrasoundDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.transform  = transform

        self.images = sorted(self.images_dir.glob('*.png'))
        self.masks  = sorted(self.masks_dir.glob('*.png'))

        if len(self.images) == 0:
            raise FileNotFoundError(f"No PNGs found in {self.images_dir.absolute()}")
        if len(self.images) != len(self.masks):
            raise ValueError(f"Image/mask count mismatch: {len(self.images)} vs {len(self.masks)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image    = Image.open(self.images[index]).convert("RGB")
        mask_rgb = Image.open(self.masks[index]).convert("RGB")

        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask_tensor = self.transform(mask_rgb)
        else:
            image       = transforms.ToTensor()(image)
            mask_tensor = transforms.ToTensor()(mask_rgb)

        combined_mask = extract_masks(mask_tensor)
        return {'image': image, 'mask': combined_mask}


class SubsetWithTransform(Dataset):
    """
    Wraps a Subset and applies transforms without mutating the shared
    parent dataset — avoids train/test transform bleed.
    """
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

        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask_tensor = self.transform(mask_rgb)
        else:
            image       = transforms.ToTensor()(image)
            mask_tensor = transforms.ToTensor()(mask_rgb)

        combined_mask = extract_masks(mask_tensor)
        return {'image': image, 'mask': combined_mask}


def extract_masks(mask_tensor):
    r = mask_tensor[0]
    g = mask_tensor[1]
    b = mask_tensor[2]

    tol = 0.05

    # LV: RGB(0, 128, 0) -> normalized (0.0, 0.502, 0.0)
    lv_mask = (
        (r < tol) &
        (g > 0.502 - tol) & (g < 0.502 + tol) &
        (b < tol)
    ).float()

    # Brain: any non-black pixel that is NOT LV
    brain_mask = (
        (mask_tensor.max(dim=0)[0] > 0.05) & (~lv_mask.bool())
    ).float()

    return torch.stack([lv_mask, brain_mask], dim=0)  # [2, H, W]


def load_ultrasound_data(images_dir="images", masks_dir="masks", batch_size=4):
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

    train_size = int(0.8 * len(full_dataset))
    test_size  = len(full_dataset) - train_size

    train_subset, test_subset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds = SubsetWithTransform(train_subset, train_transform)
    test_ds  = SubsetWithTransform(test_subset,  test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Dataset: {len(full_dataset)} total | {train_size} train | {test_size} test")
    return train_loader, test_loader