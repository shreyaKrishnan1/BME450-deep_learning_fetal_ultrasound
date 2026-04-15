import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.data_loader import load_ultrasound_data

def train_model(model, train, device, epochs=50):
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"Training in progress on {device}")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        loop = tqdm(train, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            

def test_model(model, test, device):
    model.eval()
    total_dice = 0
    
    print("Testing in progress")
    with torch.no_grad():
        for batch in test:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

            inter = (preds * masks).sum()
            dice = (2. * inter) / (preds.sum() + masks.sum() + 1e-8)
            total_dice += dice.item()

    avg_dice = total_dice / len(test)
    print(f"Final Test Dice Score: {avg_dice:.4f}")

    num_to_show = min(len(images), 3)
    fig, axes = plt.subplots(num_to_show, 3, figsize=(12, 4 * num_to_show))
    
    for i in range(num_to_show):
        axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title("Ultrasound")
        axes[i, 1].imshow(masks[i].cpu().squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(preds[i].cpu().squeeze(), cmap="gray")
        axes[i, 2].set_title(f"Prediction (Dice Score: {avg_dice:.2f})")
        for ax in axes[i]: ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train, test = load_ultrasound_data(
        images_dir="images", 
        masks_dir="masks", 
        batch_size=4
    )

    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    ).to(device)

    train_model(model, train, device, epochs=50)
    test_model(model, test, device)