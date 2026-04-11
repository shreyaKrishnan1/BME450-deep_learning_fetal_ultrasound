from pathlib import Path
import cv2
import random

# Configuration
source_dir = Path('path/to/downloaded/zenodo')  #Folder containing your images and masks
output_dir = Path('fetal_project/data')
img_size = (224, 224)
split_ratio = 0.8  # 80% train, 20% val

# Create the directory structure
for split in ['train', 'val']:
    (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

# Pair images and masks by filename
# Assuming images are in source/images and masks in source/masks
images = sorted(list((source_dir / 'images').glob('*.png')))
masks = sorted(list((source_dir / 'masks').glob('*.png')))
dataset = list(zip(images, masks))

# Shuffle and split
random.seed(42)
random.shuffle(dataset)
split_idx = int(len(dataset) * split_ratio)

def save_files(file_pairs, split_name):
    for img_path, mask_path in file_pairs:
        # Process Image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img_res = cv2.resize(img, img_size)
        cv2.imwrite(str(output_dir / split_name / 'images' / img_path.name), img_res)
        
        # Process Mask (use INTER_NEAREST to keep labels sharp)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_res = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(output_dir / split_name / 'masks' / mask_path.name), mask_res)
        # Execute
save_files(dataset[:split_idx], 'train')
save_files(dataset[split_idx:], 'val')

print(f"Done! Dataset split into {split_idx} train and {len(dataset)-split_idx} val images.")
