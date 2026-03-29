from fastapi import FastAPI, UploadFile, File
from PIL import Image
import pytesseract
import io

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Invoice API Running", "version": "1.0"}

@app.post("/extract")
async def extract_invoice(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    text = pytesseract.image_to_string(image)
    return {
        "filename": file.filename,
        "extracted_text": text,
        "status": "success"
    }
