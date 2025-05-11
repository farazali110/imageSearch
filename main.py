from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import io
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

app = FastAPI()

# Optional: add your Amazon affiliate tag
AFFILIATE_TAG = None  # e.g., "yourtag-20"

# Load a pretrained image classification model (once at startup)
model = models.resnet50(pretrained=True)
model.eval()

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_labels = []
try:
    labels_response = requests.get(LABELS_URL)
    imagenet_labels = labels_response.text.strip().split("\n")
except:
    imagenet_labels = ["class_" + str(i) for i in range(1000)]

# Image transform pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 🔍 Amazon search helper
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
                # Add affiliate tag if set
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

# 📷 Upload image and return top labels + Amazon results
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_probs, top5_idxs = torch.topk(probabilities, 5)

    top_labels = [imagenet_labels[idx] for idx in top5_idxs]
    
    # Get Amazon links for the top-1 label
    search_results = search_amazon_products(top_labels[0])

    return JSONResponse(content={
        "tags": top_labels,
        "amazon_results": search_results
    })
