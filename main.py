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


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        label = get_best_clip_label(image)
        products = search_amazon_products(label)
        return JSONResponse({
            "detected_label": label,
            "products": products
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
