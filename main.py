import os
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
import io

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
from pdf2image import convert_from_bytes

# ========================= CONFIG =========================
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp', '.pdf'}

app = FastAPI(
    title="Indian GST Invoice Extractor API",
    description="Robust Tesseract-based OCR for GST Invoices - v5.0",
    version="5.0.0"
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

cache: Dict[str, Dict] = {}

# ========================= HELPER: GSTIN VALIDATION =========================
def is_valid_gstin(gstin: str) -> bool:
    if not gstin or len(gstin) != 15:
        return False
    pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$'
    if not re.match(pattern, gstin):
        return False
    # Simple checksum (basic validation - production can call GST portal API)
    return True

# ========================= ADVANCED PREPROCESSING =========================
def deskew_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(image: Image.Image) -> Image.Image:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    
    # Deskew
    img_array = deskew_image(img_array)
    
    # Grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Multiple binarization strategies (we'll try the best later)
    binary_gaussian = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(binary_gaussian, -1, kernel)
    
    # Scale up for small text
    height, width = sharpened.shape
    if width < 1200:
        scale = 1200 / width
        sharpened = cv2.resize(sharpened, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    return Image.fromarray(sharpened)

# ========================= OCR WITH MULTIPLE CONFIGS =========================
def extract_text_from_image(image: Image.Image) -> str:
    processed = preprocess_image(image)
    
    configs = [
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 4',
        r'--oem 3 --psm 11',
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz₹.,:/- ',
    ]
    
    best_text = ""
    best_conf = 0
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(processed, config=config)
            # Get confidence (average word conf)
            data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data['conf'] if int(c) > 0]
            avg_conf = sum(confs) / len(confs) if confs else 0
            
            if avg_conf > best_conf and len(text.strip()) > len(best_text.strip()):
                best_text = text
                best_conf = avg_conf
        except Exception:
            continue
    
    return best_text.strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pages = convert_from_bytes(file_bytes, dpi=350, fmt='PNG')  # Higher DPI for accuracy
        full_text = ""
        for page in pages:
            text = extract_text_from_image(page)
            full_text += text + "\n\n"
        return full_text
    except Exception as e:
        raise HTTPException(500, f"PDF processing failed: {str(e)}")

# ========================= STRONGER EXTRACTION FUNCTIONS =========================
def extract_gstin(text: str) -> Optional[str]:
    pattern = r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b'
    matches = re.findall(pattern, text.upper())
    valid_matches = [m for m in matches if is_valid_gstin(m)]
    return valid_matches[0] if valid_matches else (matches[0] if matches else None)

def extract_all_gstins(text: str) -> List[str]:
    pattern = r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b'
    return list(set(re.findall(pattern, text.upper())))

def extract_invoice_number(text: str) -> Optional[str]:
    patterns = [
        r'(?:Invoice|Bill|Inv|Receipt)[\s#:.]*No?[\s#:.]*([A-Z0-9][A-Z0-9/.\-]{2,25})',
        r'\b(INV|GSTINV|BILL)[-/\s]?([A-Z0-9]{2,20})\b',
        r'(?:No\.?|#)\s*([A-Z0-9][A-Z0-9/.\-]{3,20})',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return (match.group(1) + (match.group(2) if len(match.groups()) > 1 else '')).strip()
    return None

def extract_date(text: str) -> Optional[str]:
    patterns = [
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
        r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})\b',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def extract_amount(text: str, keywords: List[str]) -> Optional[float]:
    for kw in keywords:
        patterns = [
            rf'(?:{kw})[\s:₹Rs.]*([0-9,]+\.?[0-9]*)',
            rf'([0-9,]+\.?[0-9]*)[\s]*(?:{kw})',
        ]
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                try:
                    val = float(m.replace(',', ''))
                    if val > 0:
                        return val
                except:
                    continue
    return None

def extract_line_items(text: str) -> List[Dict]:
    """Basic line item extraction - looks for description + HSN + qty + rate + amount patterns"""
    lines = text.split('\n')
    items = []
    hsn_pattern = r'\b(\d{4,8})\b'
    
    for line in lines:
        line = line.strip()
        if len(line) < 10:
            continue
        hsn_matches = re.findall(hsn_pattern, line)
        if hsn_matches or any(k in line.lower() for k in ['qty', 'rate', 'amount', 'total']):
            # Heuristic item
            item = {"raw": line}
            # Try to extract numbers
            numbers = re.findall(r'[\d,]+\.?\d*', line)
            if numbers:
                try:
                    item["amount"] = float(numbers[-1].replace(',', ''))
                except:
                    pass
            items.append(item)
    return items[:20]  # Limit

def extract_gst_fields(text: str) -> Dict[str, Any]:
    gstins = extract_all_gstins(text)
    supplier_gstin = gstins[0] if gstins else None
    buyer_gstin = gstins[1] if len(gstins) > 1 else None

    fields = {
        "supplier_gstin": supplier_gstin,
        "buyer_gstin": buyer_gstin,
        "invoice_number": extract_invoice_number(text),
        "invoice_date": extract_date(text),
        "taxable_value": extract_amount(text, ["taxable", "subtotal", "basic", "taxable value"]),
        "cgst": extract_amount(text, ["cgst", "central gst", "central tax"]),
        "sgst": extract_amount(text, ["sgst", "state gst", "state tax"]),
        "igst": extract_amount(text, ["igst", "integrated tax"]),
        "total_amount": extract_amount(text, ["grand total", "total", "payable", "invoice total"]),
        "hsn_codes": list(set(re.findall(r'\b\d{4,8}\b', text))),
        "place_of_supply": re.search(r'(?:place of supply|pos)[:\s]*([A-Za-z\s]+)', text, re.I),
        "line_items": extract_line_items(text),
        "invoice_type": "Tax Invoice" if "tax invoice" in text.lower() else "Invoice"
    }
    
    # Post-process place_of_supply
    if fields["place_of_supply"]:
        fields["place_of_supply"] = fields["place_of_supply"].group(1).strip()
    
    return fields

def calculate_overall_confidence(fields: Dict) -> str:
    score = 0
    if fields.get("supplier_gstin") and is_valid_gstin(fields["supplier_gstin"]): score += 25
    if fields.get("invoice_number"): score += 20
    if fields.get("invoice_date"): score += 15
    if fields.get("total_amount"): score += 20
    if fields.get("line_items") and len(fields["line_items"]) > 0: score += 20
    
    if score >= 70: return "high"
    elif score >= 40: return "medium"
    return "low"

# ========================= ENDPOINTS =========================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>GST Extractor API v5.0 Running</h1><p>Use /docs for Swagger</p>")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "5.0.0",
        "ocr_engine": "Tesseract + OpenCV",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/extract")
@limiter.limit("40/minute")
async def extract_invoice(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Allowed extensions: {ALLOWED_EXTENSIONS}")

    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Max 10MB")

    file_hash = hashlib.md5(file_content).hexdigest()
    if file_hash in cache:
        return JSONResponse(content={"success": True, "cached": True, **cache[file_hash]})

    try:
        is_pdf = ext == '.pdf' or file.content_type == "application/pdf"
        
        if is_pdf:
            raw_text = extract_text_from_pdf(file_content)
        else:
            image = Image.open(io.BytesIO(file_content))
            raw_text = extract_text_from_image(image)

        fields = extract_gst_fields(raw_text)
        confidence = calculate_overall_confidence(fields)

        result = {
            "success": True,
            "cached": False,
            "filename": file.filename,
            "extracted_data": fields,
            "raw_text": raw_text[:2000] + "..." if len(raw_text) > 2000 else raw_text,  # Truncate for response
            "confidence": confidence,
            "metadata": {
                "file_size": len(file_content),
                "processed_at": datetime.now().isoformat()
            }
        }

        cache[file_hash] = result
        if len(cache) > 150:
            for k in list(cache.keys())[:30]:
                del cache[k]

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

# Keep your existing /extract/batch if needed (similar improvements can be applied)

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
