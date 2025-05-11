from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pretrained image classification model
model = models.resnet50(pretrained=True)
model.eval()

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
import requests
labels = requests.get(LABELS_URL).text.splitlines()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, indices = torch.topk(outputs, 5)
        predicted_labels = [labels[idx] for idx in indices[0]]

    # Mock Amazon product response with affiliate links
    amazon_results = []
    for label in predicted_labels:
        amazon_results.append({
            "title": f"Amazon product for {label}",
            "product_url": f"https://www.amazon.com/s?k={label.replace(' ', '+')}&tag=your-affiliate-tag"
        })

    return {
        "tags": predicted_labels,
        "amazon_results": amazon_results
    }
