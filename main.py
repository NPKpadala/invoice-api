from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import io
import os
import re

app = FastAPI(
    title="Indian GST Invoice Extractor",
    description="Extract structured data from Indian GST invoices",
    version="2.0"
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

def preprocess_image(image):
    image = image.convert('L')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    return image

def extract_gst_fields(text):
    fields = {}
    gstin = re.findall(r'[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}', text)
    if gstin:
        fields['gstin'] = gstin[0]
    invoice = re.findall(r'(?:invoice|bill|inv)[\s#:no.]*([A-Z0-9/-]+)', text, re.IGNORECASE)
    if invoice:
        fields['invoice_number'] = invoice[0]
    date = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text)
    if date:
        fields['date'] = date[0]
    amount = re.findall(r'(?:total|amount|grand total)[\s:₹Rs.]*([0-9,]+\.?[0-9]*)', text, re.IGNORECASE)
    if amount:
        fields['total_amount'] = amount[0]
    cgst = re.findall(r'(?:cgst)[\s:@%0-9]*₹?([0-9,]+\.?[0-9]*)', text, re.IGNORECASE)
    if cgst:
        fields['cgst'] = cgst[0]
    sgst = re.findall(r'(?:sgst)[\s:@%0-9]*₹?([0-9,]+\.?[0-9]*)', text, re.IGNORECASE)
    if sgst:
        fields['sgst'] = sgst[0]
    return fields

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/extract")
async def extract_invoice(file: UploadFile = File(...)):
    try:
        allowed = ['image/jpeg','image/png','image/jpg','image/tiff','image/bmp']
        if file.content_type not in allowed:
            raise HTTPException(status_code=400, detail="Only image files supported")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed = preprocess_image(image)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed, lang='eng', config=custom_config)
        gst_fields = extract_gst_fields(text)
        return {
            "status": "success",
            "filename": file.filename,
            "extracted_fields": gst_fields,
            "raw_text": text,
            "confidence": "high" if gst_fields else "low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0"}
