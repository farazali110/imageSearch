from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# You can expand this list with product-related words
predefined_labels = [
    "white sneakers", "running shoes", "leather bag", "black boots", "sports shoes",
    "Adidas shoes", "Nike shoes", "hoodie", "jacket", "t-shirt", "smartwatch", "jeans"
]

def get_best_clip_label(image: Image.Image):
    inputs = processor(text=predefined_labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    best_idx = torch.argmax(probs).item()
    return predefined_labels[best_idx]
