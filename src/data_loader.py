import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pathlib import Path

class UltrasoundDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        self.images = sorted(list(self.images_dir.glob('*.png')))
        self.masks = sorted(list(self.masks_dir.glob('*.png')))

        if len(self.images) == 0:
            raise FileNotFoundError(f"No PNGs found in {self.images_dir.absolute()}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask_rgb = Image.open(self.masks[index]).convert("RGB")

        if self.transform:
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask_tensor = self.transform(mask_rgb)

        red_ch = mask_tensor[0, :, :]
        green_ch = mask_tensor[1, :, :]
        
        target_mask = (green_ch > red_ch).float()
        
        target_mask = target_mask.unsqueeze(0)

        return {'image': image, 'mask': target_mask}

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
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_ds.dataset.transform = train_transform
    test_ds.dataset.transform = test_transform
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader