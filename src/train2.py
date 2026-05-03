import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from data_loader    import load_ultrasound_data
from data_loader_brain import load_brain_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. METRICS
def calculate_metrics(preds, targets, smooth=1e-6):
    # preds, targets: [1, H, W]
    p = preds.view(-1)
    t = targets.view(-1)
    dice     = ((2. * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)).item()
    accuracy = (p == t).float().mean().item()
    return dice, accuracy


# 2. MODELS
def get_lv_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="swsl",
        in_channels=3,
        classes=1
    ).to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.encoder.layer4.parameters():
        param.requires_grad = True
    return model


def get_brain_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="swsl",
        in_channels=3,
        classes=1
    ).to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.encoder.layer4.parameters():
        param.requires_grad = True
    return model


# 3. TRAINING LOOP
def train_model(model, loader, epochs=30, label="Model"):
    loss_array     = []
    dice_array     = []
    accuracy_array = []

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = smp.losses.DiceLoss(mode='binary')

    print(f"\n--- Training {label} on {device} ---")
    for epoch in range(epochs):
        model.train()
        running_loss, running_dice, running_acc = 0.0, 0.0, 0.0

        for batch in loader:
            inputs = batch['image'].to(device)         # [B, 3, H, W]
            labels = batch['mask'].float().to(device)  # [B, 1, H, W]

            optimizer.zero_grad()
            outputs = model(inputs)                    # [B, 1, H, W] logits
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                dice, acc = calculate_metrics(preds[0], labels[0])
                running_dice += dice
                running_acc  += acc

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        avg_dice = running_dice / len(loader)
        avg_acc  = running_acc  / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | Acc: {avg_acc:.4f}")
        loss_array.append(avg_loss)
        dice_array.append(avg_dice)
        accuracy_array.append(avg_acc)

    return loss_array, dice_array, accuracy_array


# 4. EVALUATION & VISUALIZATION
def evaluate_and_visualize(lv_model, brain_model, loader_lv, loader_brain):
    lv_model.eval()
    brain_model.eval()

    print("\n" + "="*85)
    print(f"{'Sample':<8} | {'Dice LV':<8} | {'Acc LV':<8} | {'Dice Br':<8} | {'Acc Br':<8} | {'Rel Area %'}")
    print("-"*85)

    with torch.no_grad():
        count = 0
        for batch_lv, batch_brain in zip(loader_lv, loader_brain):
            images      = batch_lv['image'].to(device)
            lv_masks    = batch_lv['mask'].float().to(device)     # [B, 1, H, W]
            brain_masks = batch_brain['mask'].float().to(device)  # [B, 1, H, W]

            lv_preds    = (torch.sigmoid(lv_model(images))    > 0.5).float()
            brain_preds = (torch.sigmoid(brain_model(images)) > 0.5).float()

            for i in range(images.size(0)):
                count += 1
                d_lv,  acc_lv = calculate_metrics(lv_preds[i],    lv_masks[i])
                d_br,  acc_br = calculate_metrics(brain_preds[i], brain_masks[i])
                a_lv  = lv_preds[i].sum().item()
                a_br  = brain_preds[i].sum().item()
                rel_a = (a_lv / (a_br + 1e-6)) * 100

                print(f"Img {count:02d}   | {d_lv:.4f}   | {acc_lv:.4f}   | {d_br:.4f}   | {acc_br:.4f}   | {rel_a:.2f}%")

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                axes[0].imshow(img)
                axes[0].set_title("Input Ultrasound")

                axes[1].imshow(img)
                axes[1].imshow(brain_masks[i, 0].cpu(), cmap='Blues', alpha=0.3)
                axes[1].imshow(lv_masks[i,    0].cpu(), cmap='Reds',  alpha=0.5)
                axes[1].set_title("GT (Red=LV, Blue=Brain)")

                axes[2].imshow(img)
                axes[2].imshow(brain_preds[i, 0].cpu(), cmap='Blues', alpha=0.3)
                axes[2].imshow(lv_preds[i,    0].cpu(), cmap='Reds',  alpha=0.5)
                axes[2].set_title(f"LV Dice {d_lv:.2f} | Acc {acc_lv:.2f}\nRel Area: {rel_a:.1f}%")

                for ax in axes:
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(f"prediction_{count:03d}.png", dpi=100, bbox_inches='tight')
                plt.close(fig)


# --- MAIN ---
if __name__ == "__main__":
    lv_train,    lv_test    = load_ultrasound_data(   images_dir="images", masks_dir="masks", batch_size=4)
    brain_train, brain_test = load_brain_data(images_dir="images", masks_dir="masks", batch_size=4)

    lv_model    = get_lv_model()
    brain_model = get_brain_model()

    lv_losses,    lv_dices,    lv_accs    = train_model(lv_model,    lv_train,    epochs=30, label="LV Model")
    brain_losses, brain_dices, brain_accs = train_model(brain_model, brain_train, epochs=30, label="Brain Model")

    print("\nEvaluating on Test Set:")
    evaluate_and_visualize(lv_model, brain_model, lv_test, brain_test)

    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(lv_losses);    axes[0, 0].set_title("LV Loss");        axes[0, 0].set_xlabel("Epoch")
    axes[0, 1].plot(lv_dices);     axes[0, 1].set_title("LV Dice");        axes[0, 1].set_xlabel("Epoch")
    axes[0, 2].plot(lv_accs);      axes[0, 2].set_title("LV Accuracy");    axes[0, 2].set_xlabel("Epoch")
    axes[1, 0].plot(brain_losses); axes[1, 0].set_title("Brain Loss");     axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].plot(brain_dices);  axes[1, 1].set_title("Brain Dice");     axes[1, 1].set_xlabel("Epoch")
    axes[1, 2].plot(brain_accs);   axes[1, 2].set_title("Brain Accuracy"); axes[1, 2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=100, bbox_inches='tight')
    plt.show()