import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from data_loader import load_ultrasound_data

# 1. SETUP DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. DICE SCORE CALCULATION
def calculate_dice(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

import torch
import torch.nn as nn

class SmallUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SmallUNet, self).__init__()
        
        # Helper: Double Conv
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # Encoder
        self.enc1 = double_conv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = double_conv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = double_conv(32, 64)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = double_conv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = double_conv(32, 16)
        
        # Final
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        b = self.bottleneck(self.pool2(e2))
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)


# 3. MODEL SETUP
def get_model():
    model = SmallUNet().to(device)
    print(model)
    return model

# 4. TRAINING LOOP
def train_model(model, loader, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = smp.losses.DiceLoss(mode='binary')
    
    model.train()
    print(f"--- Training Started on {device} ---")
    for epoch in range(epochs):
        running_loss, running_dice = 0.0, 0.0
        for batch in loader:
            inputs, labels = batch['image'].to(device), batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate Dice for progress tracking
            preds = (torch.sigmoid(outputs) > 0.3).float()
            running_loss += loss.item()
            running_dice += calculate_dice(preds, labels).item()
        
        epoch_loss = running_loss / len(loader)
        epoch_dice = running_dice / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f}")

# 5. EVALUATION (Collects all samples for visualization)
def evaluate_and_visualize(model, loader):
    model.eval()
    all_imgs, all_masks, all_preds, all_dices = [], [], [], []

    print("\n" + "="*40)
    print(f"{'Sample #':<10} | {'Dice Score':<12}")
    print("-" * 35)

    with torch.no_grad():
        count = 0
        for batch in loader:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            for i in range(images.size(0)):
                count += 1
                dice = calculate_dice(preds[i], masks[i]).item()
                print(f"Image {count:02d}    | {dice:.4f}")
                
                all_imgs.append(images[i].cpu())
                all_masks.append(masks[i].cpu())
                all_preds.append(preds[i].cpu())
                all_dices.append(dice)

    mean_dice = sum(all_dices) / len(all_dices)
    print("-" * 35)
    print(f"MEAN DICE: {mean_dice:.4f}\n")

    # Visualization
    num_samples = len(all_imgs)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    
    if num_samples == 1: axes = axes[None, :]

    for i in range(num_samples):
        img = all_imgs[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) # Prevent div by zero
        
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(all_masks[i].squeeze(), cmap='gray')
        axes[i, 2].imshow(all_preds[i].squeeze(), cmap='viridis')
        axes[i, 2].set_title(f"Dice: {all_dices[i]:.4f}")
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    train_loader, test_loader = load_ultrasound_data(
        images_dir="images", 
        masks_dir="masks", 
        batch_size=4
    )

    model_extractor = get_model()
    train_model(model_extractor, train_loader, epochs=20)

    print("Evaluating Model Performance on Test Set:")
    evaluate_and_visualize(model_extractor, test_loader)