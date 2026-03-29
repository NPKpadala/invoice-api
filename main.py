import os
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from google import genai
from google.genai import types
from PIL import Image
import io

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp', '.pdf'}
MODEL_NAME = "gemini-2.0-flash-lite"
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Indian GST Invoice Extractor API",
    description="Extract GST data from invoices using Google Gemini AI",
    version="3.0.0"
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

cache = {}

EXTRACTION_PROMPT = """
You are a specialized Indian GST invoice parser.
Extract the following fields from this invoice:

1. gstin - 15 character GSTIN of seller
2. invoice_number - Invoice number
3. invoice_date - Date in DD/MM/YYYY format
4. vendor_name - Seller/supplier name
5. vendor_gstin - Seller GSTIN
6. buyer_name - Buyer/recipient name
7. buyer_gstin - Buyer GSTIN if available
8. place_of_supply - State name
9. taxable_amount - Amount before tax (number only)
10. cgst - CGST amount (number only)
11. sgst - SGST amount (number only)
12. igst - IGST amount if applicable (number only)
13. total_amount - Final total (number only)
14. invoice_type - Tax Invoice or Bill of Supply
15. hsn_codes - List of HSN/SAC codes found

Return ONLY valid JSON with these exact keys.
Use null for missing fields.
Amounts must be numbers not strings.

Example:
{
    "gstin": "27AAACA1234A1Z5",
    "invoice_number": "INV-2024-001",
    "invoice_date": "15/03/2024",
    "vendor_name": "ABC Enterprises Pvt Ltd",
    "vendor_gstin": "27AAACA1234A1Z5",
    "buyer_name": "XYZ Retail",
    "buyer_gstin": "27BBBCD5678B2Z3",
    "place_of_supply": "Maharashtra",
    "taxable_amount": 10000.00,
    "cgst": 900.00,
    "sgst": 900.00,
    "igst": null,
    "total_amount": 11800.00,
    "invoice_type": "Tax Invoice",
    "hsn_codes": ["9988", "9989"]
}
"""

def detect_file_type(filename: str, file_bytes: bytes) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext == '.pdf' or file_bytes.startswith(b'%PDF'):
        return 'application/pdf'
    elif ext in ['.jpg', '.jpeg'] or file_bytes.startswith(b'\xff\xd8'):
        return 'image/jpeg'
    elif ext == '.png' or file_bytes.startswith(b'\x89PNG'):
        return 'image/png'
    elif ext in ['.tiff', '.tif']:
        return 'image/tiff'
    elif ext == '.bmp':
        return 'image/bmp'
    elif ext == '.webp':
        return 'image/webp'
    else:
        return 'image/jpeg'

def preprocess_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    max_size = 2000
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    response_text = response_text.strip()
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1)
    else:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
    result = json.loads(response_text)
    if result.get('gstin'):
        result['gstin'] = result['gstin'].upper()
    for field in ['taxable_amount', 'cgst', 'sgst', 'igst', 'total_amount']:
        if result.get(field) is not None:
            try:
                result[field] = float(result[field])
            except:
                result[field] = None
    return result

def extract_with_gemini(file_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    try:
        if mime_type == 'application/pdf':
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    EXTRACTION_PROMPT,
                    types.Part.from_bytes(
                        data=file_bytes,
                        mime_type="application/pdf"
                    )
                ]
            )
        else:
            image = preprocess_image(file_bytes)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[EXTRACTION_PROMPT, image]
            )

        return parse_gemini_response(response.text)

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse response: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini API error: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>API Running! Visit /docs</h1>")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
async def list_models():
    try:
        models = client.models.list()
        return {
            "available_models": [
                m.name for m in models
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/extract")
@limiter.limit("10/minute")
async def extract_invoice(
    request: Request,
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    file_content = await file.read()
    file_size = len(file_content)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File too large. Max 10MB"
        )

    file_hash = hashlib.md5(file_content).hexdigest()
    if file_hash in cache:
        return JSONResponse(content={
            "success": True,
            "cached": True,
            "filename": file.filename,
            "extracted_data": cache[file_hash],
            "metadata": {
                "file_size": file_size,
                "processed_at": datetime.now().isoformat()
            }
        })

    mime_type = detect_file_type(file.filename, file_content)
    extracted_data = extract_with_gemini(file_content, mime_type)

    cache[file_hash] = extracted_data
    if len(cache) > 100:
        keys = list(cache.keys())[:20]
        for key in keys:
            del cache[key]

    return JSONResponse(content={
        "success": True,
        "cached": False,
        "filename": file.filename,
        "extracted_data": extracted_data,
        "metadata": {
            "file_type": mime_type,
            "file_size": file_size,
            "processed_at": datetime.now().isoformat()
        }
    })

@app.post("/extract/batch")
@limiter.limit("5/minute")
async def extract_batch(
    request: Request,
    files: List[UploadFile] = File(...)
):
    if len(files) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 files per batch"
        )

    results = []
    for file in files:
        try:
            file_content = await file.read()
            mime_type = detect_file_type(file.filename, file_content)
            data = extract_with_gemini(file_content, mime_type)
            results.append({
                "filename": file.filename,
                "success": True,
                "data": data
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return JSONResponse(content={
        "success": True,
        "total": len(files),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results,
        "processed_at": datetime.now().isoformat()
    })

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)}
    )
