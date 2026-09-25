# ================================
# 1. IMPORTS
# ================================
from ultralytics import YOLO
import os, cv2, random, shutil
from collections import Counter

# ================================
# 2. AUGMENTATION BLOCK (ADD HERE)
# ================================
label_dir = "data/annotated/train/labels"
img_dir = "data/annotated/train/images"

counts = Counter()

for file in os.listdir(label_dir):
    with open(os.path.join(label_dir, file)) as f:
        for line in f:
            counts[int(line.split()[0])] += 1

print("Before Augmentation:", counts)

max_count = max(counts.values())
minority_classes = [k for k,v in counts.items() if v < max_count]

def augment(img):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (5,5), 0)
    if random.random() < 0.5:
        img = cv2.convertScaleAbs(img, alpha=1.3, beta=25)
    return img

new_id = 0

for lbl in os.listdir(label_dir):
    lbl_path = os.path.join(label_dir, lbl)

    with open(lbl_path) as f:
        lines = f.readlines()

    classes = [int(l.split()[0]) for l in lines]

    if any(c in minority_classes for c in classes):

        img_name = lbl.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)

        for _ in range(3):  # increase if needed
            aug_img = augment(img)

            new_img = f"aug_{new_id}.jpg"
            new_lbl = f"aug_{new_id}.txt"

            cv2.imwrite(os.path.join(img_dir, new_img), aug_img)
            shutil.copy(lbl_path, os.path.join(label_dir, new_lbl))

            new_id += 1

print("Augmented:", new_id)

# ================================
# 3. TRAIN MODEL
# ================================
model = YOLO("yolov8s.pt")   # upgraded model

model.train(
    data="data/annotated/data.yaml",
    epochs=100,
    imgsz=768,
    batch=16,
    name="improved_model"
)