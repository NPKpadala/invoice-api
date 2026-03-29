import os
import re
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_bytes
import pytesseract

app = FastAPI(title="GST Invoice Extractor")

# Mount static files (your HTML/CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment variables")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Helper: Extract text from image using Gemini Vision
def extract_text_from_image(image_bytes: bytes) -> str:
    """Send image directly to Gemini Vision API"""
    image = Image.open(io.BytesIO(image_bytes))
    prompt = """Extract ALL text from this Indian GST invoice exactly as it appears. 
    Return only the raw text, no explanations."""
    response = model.generate_content([prompt, image])
    return response.text

# Helper: Extract text from PDF (convert first page to image)
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Convert PDF to image, then use Gemini Vision"""
    images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
    if not images:
        return ""
    img_byte_arr = io.BytesIO()
    images[0].save(img_byte_arr, format='PNG')
    return extract_text_from_image(img_byte_arr.getvalue())

# Helper: Parse GST fields using Gemini (text-only)
def parse_gst_fields(text: str) -> dict:
    """Send extracted text to Gemini for structured parsing"""
    prompt = f"""
    You are a GST invoice parser. Extract these fields from the invoice text below.
    
    Fields to extract:
    - gstin (GSTIN number, format: 15 alphanumeric)
    - invoice_number (any alphanumeric)
    - date (in DD/MM/YYYY or YYYY-MM-DD format)
    - total_amount (numeric value, no currency symbol)
    - cgst (numeric value)
    - sgst (numeric value)
    
    Invoice text:
    {text[:3000]}
    
    Return ONLY valid JSON like this:
    {{"gstin": "value", "invoice_number": "value", "date": "value", "total_amount": 0.0, "cgst": 0.0, "sgst": 0.0}}
    If a field is missing, use null for string, 0.0 for numbers.
    """
    response = model.generate_content(prompt)
    # Parse JSON from response
    import json
    try:
        return json.loads(response.text)
    except:
        # Fallback regex extraction
        return {
            "gstin": re.search(r'[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z]{1}[0-9A-Z]{1}', text),
            "invoice_number": re.search(r'INV[-/ ]?[0-9]+', text),
            "date": re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', text),
            "total_amount": 0.0,
            "cgst": 0.0,
            "sgst": 0.0
        }

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the beautiful drag-and-drop UI"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/extract")
async def extract_invoice(file: UploadFile = File(...)):
    """Extract GST fields from uploaded invoice"""
    contents = await file.read()
    filename = file.filename.lower()
    
    # Route to correct extractor
    if filename.endswith('.pdf'):
        extracted_text = extract_text_from_pdf(contents)
    elif filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        extracted_text = extract_text_from_image(contents)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    if not extracted_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted")
    
    # Parse structured fields
    parsed_data = parse_gst_fields(extracted_text)
    
    return JSONResponse(content={
        "success": True,
        "filename": file.filename,
        "extracted_data": parsed_data,
        "raw_ocr_text": extracted_text[:500]  # for debugging
    })

@app.get("/health")
async def health_check():
    return {"status": "healthy", "api": "GST Extractor"}
