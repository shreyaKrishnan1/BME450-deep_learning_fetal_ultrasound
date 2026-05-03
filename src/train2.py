import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from data_loader2 import load_ultrasound_data

# 1. SETUP DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. DICE SCORE CALCULATION
def calculate_metrics(preds, targets, smooth=1e-6):
    # preds and targets are [2, H, W] (single sample, 2 channels)

    # LV is Channel 0
    pred_lv = preds[0].view(-1)
    target_lv = targets[0].view(-1)
    intersection_lv = (pred_lv * target_lv).sum()
    dice_lv = (2. * intersection_lv + smooth) / (pred_lv.sum() + target_lv.sum() + smooth)

    # Brain is Channel 1
    pred_brain = preds[1].view(-1)
    target_brain = targets[1].view(-1)
    intersection_brain = (pred_brain * target_brain).sum()
    dice_brain = (2. * intersection_brain + smooth) / (pred_brain.sum() + target_brain.sum() + smooth)

    # Pixel Accuracy (across both channels)
    correct = (preds == targets).float().sum()
    accuracy = correct / targets.numel()

    area_lv = pred_lv.sum().item()
    area_brain = pred_brain.sum().item()
    rel_area = (area_lv / (area_brain + smooth)) * 100

    return dice_lv.item(), dice_brain.item(), accuracy.item(), area_lv, rel_area

# 3. LABEL SHAPE FIX
def fix_label_shape(labels, device):
    """
    Ensures labels are [B, 2, H, W] float for multilabel Dice loss.
    Handles these incoming shapes from the data loader:
      - [B, H, W]    : integer class-index mask (0=bg, 1=LV, 2=Brain)
      - [B, 1, H, W] : single-channel class-index mask
      - [B, 2, H, W] : already correct binary masks
    """
    if labels.ndim == 3:
        # [B, H, W] -> one-hot [B, 2, H, W]
        # Assumes pixel values: 1 = LV (ch0), 2 = Brain (ch1)
        B, H, W = labels.shape
        lv    = (labels == 1).unsqueeze(1).float()   # [B, 1, H, W]
        brain = (labels == 2).unsqueeze(1).float()   # [B, 1, H, W]
        return torch.cat([lv, brain], dim=1).to(device)

    elif labels.ndim == 4 and labels.shape[1] == 1:
        # [B, 1, H, W] -> one-hot [B, 2, H, W]
        labels = labels.squeeze(1).long()            # [B, H, W]
        B, H, W = labels.shape
        lv    = (labels == 1).unsqueeze(1).float()
        brain = (labels == 2).unsqueeze(1).float()
        return torch.cat([lv, brain], dim=1).to(device)

    else:
        # Already [B, 2, H, W]
        return labels.float().to(device)

# 4. MODEL SETUP
def get_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="swsl",
        in_channels=3,
        classes=2
    ).to(device)

    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.encoder.layer4.parameters():
        param.requires_grad = True

    return model

# 5. TRAINING LOOP
def train_model(model, loader, epochs=30, loss_array=None, dice_lv_array=None):
    if loss_array is None:
        loss_array = []
    if dice_lv_array is None:
        dice_lv_array = []

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = smp.losses.DiceLoss(mode='multilabel')

    print(f"--- Training Started on {device} ---")
    for epoch in range(epochs):
        model.train()
        running_loss, running_dice_lv = 0.0, 0.0

        for batch in loader:
            inputs = batch['image'].to(device)
            labels = fix_label_shape(batch['mask'], device)  # FIX: normalize shape

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                dice_lv, _, _, _, _ = calculate_metrics(preds[0], labels[0])

            running_loss += loss.item()
            running_dice_lv += dice_lv

        avg_loss = running_loss / len(loader)
        avg_dice = running_dice_lv / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LV Dice: {avg_dice:.4f}")
        loss_array.append(avg_loss)
        dice_lv_array.append(avg_dice)

    return loss_array, dice_lv_array

# 6. EVALUATION & VISUALIZATION
def evaluate_and_visualize(model, loader):
    model.eval()
    print("\n" + "="*80)
    print(f"{'Sample':<8} | {'Dice LV':<8} | {'Dice Br':<8} | {'LV Area':<10} | {'Rel Area %':<10}")
    print("-" * 80)

    with torch.no_grad():
        count = 0
        for batch in loader:
            images = batch['image'].to(device)
            masks  = fix_label_shape(batch['mask'], device)  # FIX: normalize shape
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            for i in range(images.size(0)):
                count += 1
                d_lv, d_br, acc, a_lv, rel_a = calculate_metrics(preds[i], masks[i])
                print(f"Img {count:02d}   | {d_lv:.4f} | {d_br:.4f} | {a_lv:<10.0f} | {rel_a:.2f}%")

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                axes[0].imshow(img)
                axes[0].set_title("Input Ultrasound")

                axes[1].imshow(img)
                axes[1].imshow(masks[i, 1].cpu(), cmap='Blues', alpha=0.3)  # Brain
                axes[1].imshow(masks[i, 0].cpu(), cmap='Reds',  alpha=0.5)  # LV
                axes[1].set_title("GT (Red=LV, Blue=Brain)")

                axes[2].imshow(img)
                axes[2].imshow(preds[i, 1].cpu(), cmap='Blues', alpha=0.3)
                axes[2].imshow(preds[i, 0].cpu(), cmap='Reds',  alpha=0.5)
                axes[2].set_title(f"Pred: LV Dice {d_lv:.2f} | Rel Area: {rel_a:.1f}%")

                for ax in axes:
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(f"prediction_{count:03d}.png", dpi=100, bbox_inches='tight')
                plt.close(fig)  # free memory

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    train_loader, test_loader = load_ultrasound_data(
        images_dir="images",
        masks_dir="masks",
        batch_size=4
    )

    model_extractor = get_model()
    num_epochs = 30

    loss_array, dice_lv_array = train_model(
        model_extractor, train_loader, epochs=num_epochs
    )

    print("\nEvaluating Model Performance on Test Set:")
    evaluate_and_visualize(model_extractor, test_loader)

    # Plot training curves
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), loss_array)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), dice_lv_array)
    plt.title("LV Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=100, bbox_inches='tight')
    plt.show()