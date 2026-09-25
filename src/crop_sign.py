import cv2
import os
from ultralytics import YOLO

model = YOLO("../models/detector.pt")

input_folder = "../data/raw_images"
output_folder = "../data/cropped_signs/unlabeled"

os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    results = model(img)

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]

        save_path = f"{output_folder}/{img_name}_{i}.jpg"
        cv2.imwrite(save_path, crop)

print("Cropping completed!")