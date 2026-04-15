import matplotlib.pyplot as plt
from data_loader import get_ultrasound_loaders
import torch

def view_data():
    train_loader, _ = get_ultrasound_loaders(
        images_dir="images", 
        masks_dir="masks", 
        batch_size=1
    )
    print("Displaying first 4 samples")
    plt.figure(figsize=(15, 10))
    for i, batch in enumerate(train_loader):
        if i == 4: break
        image = batch['image'][0].permute(1, 2, 0)
        mask = batch['mask'][0].squeeze()

        plt.subplot(2, 4, i + 1)
        plt.imshow(image)
        plt.title(f"Sample {i+1}: Image")
        plt.axis("off")

        plt.subplot(2, 4, i + 5)
        plt.imshow(mask, cmap="gray")
        plt.title(f"Sample {i+1}: Mask (Green Only)")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_data()