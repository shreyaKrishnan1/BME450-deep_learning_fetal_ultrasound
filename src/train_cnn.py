import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_loader import load_ultrasound_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. METRICS
def calculate_dice(preds, targets, smooth=1e-6):
    preds   = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def calculate_accuracy(preds, targets):
    return (preds.view(-1) == targets.view(-1)).float().mean().item()


# 2. 8-LAYER CNN
class SegmentationCNN(nn.Module):
    """
    8-layer encoder-decoder CNN for binary segmentation.

    Encoder (4 conv layers): progressively downsamples and extracts features
    Decoder (4 conv layers): progressively upsamples back to input resolution

    Layer breakdown:
      Layer 1: Conv 3->32,   BN, ReLU          (feature extraction)
      Layer 2: Conv 32->64,  BN, ReLU, MaxPool (downsample 1/2)
      Layer 3: Conv 64->128, BN, ReLU          (feature extraction)
      Layer 4: Conv 128->256 BN, ReLU, MaxPool (downsample 1/4)
      Layer 5: ConvT 256->128, BN, ReLU        (upsample 1/2)
      Layer 6: Conv  128->64,  BN, ReLU        (refine)
      Layer 7: ConvT 64->32,   BN, ReLU        (upsample back to full res)
      Layer 8: Conv  32->1                     (output logits)
    """
    def __init__(self):
        super().__init__()

        # --- Encoder ---
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 224 -> 112
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 112 -> 56
        )

        # --- Decoder ---
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 56 -> 112
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 112 -> 224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer8 = nn.Conv2d(32, 1, kernel_size=1)               # output logits

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x  # [B, 1, H, W] raw logits


def get_model():
    return SegmentationCNN().to(device)


# 3. LOSS
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)
        preds   = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)


# 4. TRAINING
loss_progression = []
dice_progression = []
acc_progression  = []

def train_model(model, loader, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceLoss()

    print(f"--- Training Started on {device} ---")
    for epoch in range(epochs):
        model.train()
        running_loss, running_dice, running_acc = 0.0, 0.0, 0.0

        for batch in loader:
            inputs = batch['image'].to(device)         # [B, 3, H, W]
            labels = batch['mask'].float().to(device)  # [B, 1, H, W]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                running_dice += calculate_dice(preds, labels).item()
                running_acc  += calculate_accuracy(preds, labels)

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        epoch_dice = running_dice / len(loader)
        epoch_acc  = running_acc  / len(loader)

        loss_progression.append(epoch_loss)
        dice_progression.append(epoch_dice)
        acc_progression.append(epoch_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f} | Acc: {epoch_acc:.4f}")

    total_epochs = range(1, epochs + 1)

    plt.figure(1)
    plt.plot(total_epochs, loss_progression)
    plt.title(f"Loss Progression over {epochs} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.figure(2)
    plt.plot(total_epochs, dice_progression)
    plt.title(f"Dice Progression over {epochs} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")

    plt.figure(3)
    plt.plot(total_epochs, acc_progression)
    plt.title(f"Accuracy Progression over {epochs} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel Accuracy")


# 5. EVALUATION
def evaluate_and_visualize(model, loader):
    model.eval()
    all_imgs, all_masks, all_preds, all_dices, all_accs = [], [], [], [], []

    print("\n" + "="*50)
    print(f"{'Sample #':<10} | {'Dice Score':<12} | {'Accuracy':<10}")
    print("-"*50)

    with torch.no_grad():
        count = 0
        for batch in loader:
            images = batch['image'].to(device)
            masks  = batch['mask'].float().to(device)
            outputs = model(images)
            preds   = (torch.sigmoid(outputs) > 0.5).float()

            for i in range(images.size(0)):
                count += 1
                dice = calculate_dice(preds[i], masks[i]).item()
                acc  = calculate_accuracy(preds[i], masks[i])
                print(f"Image {count:02d}    | {dice:.4f}       | {acc:.4f}")

                all_imgs.append(images[i].cpu())
                all_masks.append(masks[i].cpu())
                all_preds.append(preds[i].cpu())
                all_dices.append(dice)
                all_accs.append(acc)

    mean_dice = sum(all_dices) / len(all_dices)
    mean_acc  = sum(all_accs)  / len(all_accs)
    print("-"*50)
    print(f"MEAN DICE:     {mean_dice:.4f}")
    print(f"MEAN ACCURACY: {mean_acc:.4f}\n")

    num_samples = len(all_imgs)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        img = all_imgs[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input")
        axes[i, 1].imshow(all_masks[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(all_preds[i].squeeze(), cmap='viridis')
        axes[i, 2].set_title(f"Pred | Dice: {all_dices[i]:.4f}\nAcc: {all_accs[i]:.4f}")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# --- MAIN ---
if __name__ == "__main__":
    train_loader, test_loader = load_ultrasound_data(
        images_dir="images",
        masks_dir="masks",
        batch_size=4
    )

    model = get_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_model(model, train_loader, epochs=30)

    print("Evaluating Model Performance on Test Set:")
    evaluate_and_visualize(model, test_loader)