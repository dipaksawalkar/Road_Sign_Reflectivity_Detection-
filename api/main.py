from fastapi import FastAPI, UploadFile
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms, models
from torch import nn

app = FastAPI()

# Load models
detector = YOLO("../models/detector.pt")

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

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = detector(img)

    predictions = []

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]

        tensor = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = classifier(tensor)
            label = classes[torch.argmax(pred).item()]

        predictions.append(label)

    return {"predictions": predictions}