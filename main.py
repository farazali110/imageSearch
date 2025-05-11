from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from clip_utils import get_best_clip_label
from PIL import Image
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from urllib.parse import quote

app = FastAPI()

# Optional: put your affiliate tag here later
AFFILIATE_TAG = None  # e.g., "yourtag-20"

def search_amazon_products(query: str, max_results=5):
    search_url = f"https://www.google.com/search?q={quote(query + ' site:amazon.com')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for result in soup.select(".tF2Cxc")[:max_results]:
        title_elem = result.select_one("h3")
        link_elem = result.select_one("a")
        if title_elem and link_elem:
            url = link_elem["href"]
            if "amazon.com" in url:
                # Optional: Add affiliate tag later
                if AFFILIATE_TAG and "tag=" not in url:
                    if "?" in url:
                        url += f"&tag={AFFILIATE_TAG}"
                    else:
                        url += f"?tag={AFFILIATE_TAG}"

                results.append({
                    "title": title_elem.get_text(),
                    "product_url": url
                })
    return results

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as T
from torchvision import models
import requests

app = FastAPI()

# Load Pretrained ResNet50 Model
model = models.resnet50(pretrained=True)
model.eval()

# Image transformation
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Labels URL for ImageNet
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
labels = requests.get(LABELS_URL).json()
labels = {int(k): v[1] for k, v in labels.items()}

# Image upload and analysis endpoint
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        output = model(image_tensor)

    # Get top 5 predictions
    _, indices = torch.topk(output, 5)
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    top_labels = [(labels[idx.item()], probabilities[idx].item()) for idx in indices[0]]

    return JSONResponse(content={"tags": top_labels})
