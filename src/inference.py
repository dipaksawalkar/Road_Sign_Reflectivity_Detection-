import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
from torch import nn

# Load YOLO
detector = YOLO("../models/detector.pt")

# Load classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = models.resnet18(pretrained=False)
classifier.fc = nn.Linear(classifier.fc.in_features, 3)
classifier.load_state_dict(torch.load("../models/classifier.pth", map_location=device))
classifier = classifier.to(device)
classifier.eval()

classes = ["bad", "good", "moderate"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    img = cv2.imread(image_path)

    results = detector(img)
    outputs = []

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]

        tensor = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = classifier(tensor)
            label = classes[torch.argmax(pred).item()]

        outputs.append(label)

    return outputs

if __name__ == "__main__":
    print(predict("../data/raw_images/test.jpg"))