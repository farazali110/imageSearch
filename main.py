from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Image search backend is live"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))  # Optional image load
    return {"filename": file.filename}
