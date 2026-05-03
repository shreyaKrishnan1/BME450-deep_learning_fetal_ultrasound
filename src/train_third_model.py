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

# 3. PIXEL ACCURACY CALCULATION
def calculate_pixel_accuracy(preds, targets):
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

# 4. MODEL SETUP
# 3. MODEL SETUP
def get_model():
    model = smp.Unet(
        encoder_name="efficientnet-b0", 
        encoder_weights="advprop", 
        in_channels=3, 
        classes=1
    ).to(device)

    for param in model.encoder.parameters():
        param.requires_grad = False
    return model

loss_progression = []
dice_progression = []
pixel_acc_progression = []

# 5. TRAINING LOOP
def train_model(model, loader, epochs=30):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = smp.losses.DiceLoss(mode='binary')
    
    model.train()
    print(f"--- Training Started on {device} ---")
    for epoch in range(epochs):
        running_loss, running_dice, running_pixel_acc = 0.0, 0.0, 0.0
        for batch in loader:
            inputs, labels = batch['image'].to(device), batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_loss += loss.item()
            running_dice += calculate_dice(preds, labels).item()
            running_pixel_acc += calculate_pixel_accuracy(preds, labels)
        
        epoch_loss = running_loss / len(loader)
        epoch_dice = running_dice / len(loader)
        epoch_pixel_acc = running_pixel_acc / len(loader)

        loss_progression.append(epoch_loss)
        dice_progression.append(epoch_dice)
        pixel_acc_progression.append(epoch_pixel_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f} | Pixel Acc: {epoch_pixel_acc:.4f}")

    total_epochs = range(1, epochs + 1)

    plt.figure(1) 
    plt.plot(total_epochs, loss_progression)
    plt.title(f"Loss Progression over {epochs} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.figure(2)
    plt.plot(total_epochs, dice_progression)
    plt.title(f'DICE Progression over {epochs} Epochs')
    plt.xlabel("Epoch")
    plt.ylabel("DICE Score")

    plt.figure(3)
    plt.plot(total_epochs, pixel_acc_progression)
    plt.title(f'Pixel Accuracy Progression over {epochs} Epochs')
    plt.xlabel("Epoch")
    plt.ylabel("Pixel Accuracy")

# 6. EVALUATION
def evaluate_and_visualize(model, loader):
    model.eval()
    all_imgs, all_masks, all_preds, all_dices, all_pixel_accs = [], [], [], [], []

    print("\n" + "="*50)
    print(f"{'Sample #':<10} | {'Dice Score':<12} | {'Pixel Acc':<10}")
    print("-" * 45)

    with torch.no_grad():
        count = 0
        for batch in loader:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            for i in range(images.size(0)):
                count += 1
                dice = calculate_dice(preds[i], masks[i]).item()
                pixel_acc = calculate_pixel_accuracy(preds[i], masks[i])
                print(f"Image {count:02d}    | {dice:<12.4f} | {pixel_acc:.4f}")
                
                all_imgs.append(images[i].cpu())
                all_masks.append(masks[i].cpu())
                all_preds.append(preds[i].cpu())
                all_dices.append(dice)
                all_pixel_accs.append(pixel_acc)

    mean_dice = sum(all_dices) / len(all_dices)
    mean_pixel_acc = sum(all_pixel_accs) / len(all_pixel_accs)
    print("-" * 45)
    print(f"MEAN DICE:         {mean_dice:.4f}")
    print(f"MEAN PIXEL ACC:    {mean_pixel_acc:.4f}\n")

    # Visualization
    num_samples = len(all_imgs)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    
    if num_samples == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        img = all_imgs[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(all_masks[i].squeeze(), cmap='gray')
        axes[i, 2].imshow(all_preds[i].squeeze(), cmap='viridis')
        axes[i, 2].set_title(f"Dice: {all_dices[i]:.4f} | PA: {all_pixel_accs[i]:.4f}")
        for ax in axes[i]:
            ax.axis('off')

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
    train_model(model_extractor, train_loader, epochs=30)

    print("Evaluating Model Performance on Test Set:")
    evaluate_and_visualize(model_extractor, test_loader)