import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from data_loader import load_ultrasound_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_dice(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def get_model():
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",  # better default than advprop
        in_channels=3,
        classes=1
    ).to(device)

    # DO NOT freeze encoder
    for param in model.parameters():
        param.requires_grad = True

    return model


bce_loss = torch.nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode='binary')

def combined_loss(outputs, targets):
    return 0.5 * bce_loss(outputs, targets) + 0.5 * dice_loss(outputs, targets)



def train_model(model, train_loader, val_loader, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_dice = 0.0

    print(f"--- Training on {device} ---")

    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        train_loss, train_dice = 0.0, 0.0

        for batch in train_loader:
            inputs = batch['image'].to(device)
            labels = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = combined_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()

            train_loss += loss.item()
            train_dice += calculate_dice(preds, labels).item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # ===== VALIDATION =====
        model.eval()
        val_dice = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image'].to(device)
                labels = batch['mask'].to(device)

                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()

                val_dice += calculate_dice(preds, labels).item()

        val_dice /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Dice:   {val_dice:.4f}")
        print("-" * 40)

        # ===== SAVE BEST MODEL =====
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model!")


def evaluate_and_visualize(model, loader):
    model.eval()

    all_imgs, all_masks, all_preds, all_dices = [], [], [], []

    print("\n" + "="*40)
    print(f"{'Sample #':<10} | {'Dice Score':<12}")
    print("-" * 35)

    with torch.no_grad():
        count = 0
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            for i in range(images.size(0)):
                count += 1
                dice = calculate_dice(preds[i], masks[i]).item()

                print(f"Image {count:02d} | {dice:.4f}")

                all_imgs.append(images[i].cpu())
                all_masks.append(masks[i].cpu())
                all_preds.append(preds[i].cpu())
                all_dices.append(dice)

    mean_dice = sum(all_dices) / len(all_dices)
    print("-" * 35)
    print(f"MEAN DICE: {mean_dice:.4f}\n")

    # ===== VISUALIZATION =====
    num_samples = len(all_imgs)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))

    if num_samples == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        img = all_imgs[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")

        axes[i, 1].imshow(all_masks[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(all_preds[i].squeeze(), cmap='viridis')
        axes[i, 2].set_title(f"Pred (Dice: {all_dices[i]:.4f})")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# =========================
# 7. MAIN
# =========================
if __name__ == "__main__":
    train_loader, val_loader = load_ultrasound_data(
        images_dir="images",
        masks_dir="masks",
        batch_size=4
    )

    model = get_model()

    train_model(model, train_loader, val_loader, epochs=50)

    # Load best model before evaluation
    model.load_state_dict(torch.load("best_model.pth"))

    print("Evaluating on validation set:")
    evaluate_and_visualize(model, val_loader)

