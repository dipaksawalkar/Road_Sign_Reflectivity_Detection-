import os
import shutil
import random

# Source paths (your current data)
src_images = "data/train/images"
src_labels = "data/train/labels"

# Destination paths (target structure)
base = "data/annotated"

train_img_dst = os.path.join(base, "train/images")
train_lbl_dst = os.path.join(base, "train/labels")

val_img_dst = os.path.join(base, "valid/images")
val_lbl_dst = os.path.join(base, "valid/labels")

# Create folders
os.makedirs(train_img_dst, exist_ok=True)
os.makedirs(train_lbl_dst, exist_ok=True)
os.makedirs(val_img_dst, exist_ok=True)
os.makedirs(val_lbl_dst, exist_ok=True)

# Get all images
images = [f for f in os.listdir(src_images) if f.endswith((".jpg", ".png", ".jpeg"))]

# Shuffle images
random.shuffle(images)

# Split 80-20
split_index = int(0.8 * len(images))

train_images = images[:split_index]
val_images = images[split_index:]

# Function to move files
def move_files(image_list, img_dst, lbl_dst):
    for img in image_list:
        label = os.path.splitext(img)[0] + ".txt"

        src_img_path = os.path.join(src_images, img)
        src_lbl_path = os.path.join(src_labels, label)

        if os.path.exists(src_lbl_path):
            shutil.copy(src_img_path, os.path.join(img_dst, img))
            shutil.copy(src_lbl_path, os.path.join(lbl_dst, label))
        else:
            print(f"⚠️ Label missing for {img}")

# Move data
move_files(train_images, train_img_dst, train_lbl_dst)
move_files(val_images, val_img_dst, val_lbl_dst)

print("✅ Dataset prepared successfully!")
print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")