import os
import re
import json
import hashlib
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, status
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import google.generativeai as genai
from PIL import Image
import io
import magic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_MIME_TYPES = {
        'image/jpeg', 'image/png', 'image/tiff', 
        'image/bmp', 'image/webp', 'application/pdf'
    }
    RATE_LIMIT = "10/minute"  # 10 requests per minute on free tier
    REQUEST_TIMEOUT = 30  # seconds
    GEMINI_MODEL = "gemini-1.5-flash"
    CACHE_TTL = 86400  # 24 hours

# Initialize FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    if not Config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment variables")
    
    genai.configure(api_key=Config.GEMINI_API_KEY)
    app.state.model = genai.GenerativeModel('gemini-1.5-flash')
    print(f"✅ {Config.GEMINI_MODEL} initialized successfully")
    print(f"✅ API ready on Render.com free tier")
    yield
    # Shutdown
    print("👋 Shutting down...")

app = FastAPI(
    title="Indian GST Invoice Extractor API",
    description="Extract GST data from invoices using Google Gemini AI",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Restrict in production
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory cache (simple, replace with Redis for production)
cache = {}

# Helper Functions
class GSTValidator:
    """GSTIN validation utilities"""
    
    @staticmethod
    def validate_gstin(gstin: str) -> bool:
        """Validate Indian GSTIN format"""
        if not gstin or len(gstin) != 15:
            return False
        pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z]{1}[0-9A-Z]{1}$'
        return bool(re.match(pattern, str(gstin).upper()))
    
    @staticmethod
    def extract_gstin(text: str) -> Optional[str]:
        """Extract GSTIN from text using regex"""
        pattern = r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z]{1}[0-9A-Z]{1}\b'
        match = re.search(pattern, text.upper())
        return match.group(0) if match else None

class AmountParser:
    """Parse Indian currency amounts"""
    
    @staticmethod
    def parse_amount(text: str) -> float:
        """Convert Indian currency string to float"""
        if not text:
            return 0.0
        # Remove ₹ symbol, commas, spaces
        cleaned = re.sub(r'[₹,\s]', '', str(text))
        # Extract numbers and decimal
        match = re.search(r'(\d+(?:\.\d{2})?)', cleaned)
        if match:
            return float(match.group(1))
        return 0.0

class InvoiceExtractor:
    """Main invoice extraction logic"""
    
    @staticmethod
    async def detect_file_type(file_bytes: bytes) -> str:
        """Detect MIME type using magic numbers"""
        try:
            mime = magic.from_buffer(file_bytes, mime=True)
            return mime
        except:
            # Fallback to binary inspection for common types
            if file_bytes.startswith(b'%PDF'):
                return 'application/pdf'
            elif file_bytes.startswith(b'\xff\xd8'):
                return 'image/jpeg'
            elif file_bytes.startswith(b'\x89PNG'):
                return 'image/png'
            else:
                raise HTTPException(status_code=400, detail="Unknown file type")
    
    @staticmethod
    async def preprocess_image(image_bytes: bytes) -> bytes:
        """Preprocess image for better OCR results"""
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        # Resize if too large (max 2000px)
        max_size = 2000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        # Save to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG', optimize=True)
        return img_byte_arr.getvalue()
    
    @staticmethod
    async def extract_with_gemini(model, file_bytes: bytes, mime_type: str) -> Dict[str, Any]:
        """Extract invoice data using Gemini Vision API"""
        
        prompt = """
You are a specialized Indian GST invoice parser. Extract the following fields from the invoice image/PDF:

1. gstin (15-character GSTIN, format: XXAAAAA0000A0Z0)
2. invoice_number (alphanumeric, usually INV-xxxx or similar)
3. invoice_date (in DD/MM/YYYY format)
4. vendor_name (supplier/seller name)
5. vendor_gstin (supplier's GSTIN)
6. buyer_name (recipient/buyer name)
7. buyer_gstin (recipient's GSTIN, if available)
8. place_of_supply (state name where goods/services supplied)
9. taxable_amount (total before tax, numeric only)
10. cgst (Central GST amount, numeric)
11. sgst (State GST amount, numeric)
12. igst (Integrated GST amount, numeric if applicable)
13. cess (Cess amount, if applicable)
14. total_amount (final total including all taxes)
15. invoice_type (Tax Invoice, Bill of Supply, etc.)

Return ONLY valid JSON with these exact keys. Use null for missing fields. Amounts should be numbers, not strings.
Example response format:
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
    "cess": null,
    "total_amount": 11800.00,
    "invoice_type": "Tax Invoice"
}
"""
        
        try:
            # For PDF, we need to handle differently
            if mime_type == 'application/pdf':
                # Gemini can handle PDF directly
                response = model.generate_content([
                    prompt,
                    {"mime_type": "application/pdf", "data": file_bytes}
                ])
            else:
                # For images, preprocess first
                processed_bytes = await InvoiceExtractor.preprocess_image(file_bytes)
                response = model.generate_content([
                    prompt,
                    {"mime_type": "image/png", "data": processed_bytes}
                ])
            
            # Parse JSON response
            response_text = response.text.strip()
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            else:
                # Try direct JSON parsing
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
            
            result = json.loads(response_text)
            
            # Post-process and validate
            if result.get('gstin'):
                result['gstin'] = result['gstin'].upper()
            
            # Ensure numeric fields are floats
            for field in ['taxable_amount', 'cgst', 'sgst', 'igst', 'cess', 'total_amount']:
                if result.get(field) is not None:
                    try:
                        result[field] = float(result[field])
                    except:
                        result[field] = None
            
            return result
            
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse Gemini response as JSON: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Gemini API error: {str(e)}"
            )

# API Endpoints
@app.get("/", response_class=HTMLResponse)
@limiter.limit("100/minute")
async def serve_frontend(request: Request):
    """Serve the modern drag-and-drop UI"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>GST Invoice Extractor</title></head>
        <body>
            <h1>Indian GST Invoice Extractor API</h1>
            <p>API is running! Visit <a href="/docs">/docs</a> for API documentation.</p>
            <p>Upload invoices using POST /extract</p>
        </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "service": "Indian GST Invoice Extractor",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "cache_size": len(cache)
    }

@app.post("/extract")
@limiter.limit(Config.RATE_LIMIT)
async def extract_invoice(
    request: Request,
    file: UploadFile = File(..., description="Invoice file (PDF or image)")
):
    """
    Extract GST data from an invoice file.
    
    - **file**: PDF or image (JPG, PNG, TIFF, BMP, WEBP) up to 10MB
    - **Returns**: Structured JSON with all GST fields
    """
    
    # Validate file presence
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Read file content
    file_content = await file.read()
    file_size = len(file_content)
    
    # Validate file size
    if file_size > Config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max {Config.MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Validate file type
    try:
        mime_type = await InvoiceExtractor.detect_file_type(file_content)
        if mime_type not in Config.ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {mime_type}. Allowed: PDF, JPG, PNG, TIFF, BMP, WEBP"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}")
    
    # Check cache
    file_hash = hashlib.md5(file_content).hexdigest()
    if file_hash in cache:
        return JSONResponse(content={
            "success": True,
            "cached": True,
            "filename": file.filename,
            "extracted_data": cache[file_hash],
            "metadata": {
                "file_type": mime_type,
                "file_size": file_size,
                "processed_at": datetime.now().isoformat()
            }
        })
    
    try:
        # Extract data using Gemini
        extracted_data = await InvoiceExtractor.extract_with_gemini(
            app.state.model,
            file_content,
            mime_type
        )
        
        # Cache result
        cache[file_hash] = extracted_data
        
        # Clean up cache if too large
        if len(cache) > 100:
            # Remove oldest 20 entries
            keys_to_remove = list(cache.keys())[:20]
            for key in keys_to_remove:
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

@app.post("/extract/batch")
@limiter.limit("5/minute")
async def extract_batch_invoices(
    request: Request,
    files: List[UploadFile] = File(..., description="Multiple invoice files (max 5)")
):
    """
    Extract data from multiple invoices in one request.
    
    - **files**: Up to 5 invoice files (PDF or images)
    - **Returns**: Array of extraction results
    """
    
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files per batch request")
    
    results = []
    for file in files:
        try:
            # Create a mock request-like object for rate limiting
            result = await extract_invoice(request, file)
            results.append({
                "filename": file.filename,
                "success": True,
                "data": result.body
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "total_files": len(files),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results,
        "processed_at": datetime.now().isoformat()
    })

@app.get("/stats")
@limiter.limit("10/minute")
async def get_stats(request: Request):
    """Get API usage statistics"""
    return {
        "cache_hits": len(cache),
        "cache_size_mb": sum(len(str(v)) for v in cache.values()) / (1024 * 1024),
        "max_cache_size": 100,
        "status": "operational",
        "gemini_model": Config.GEMINI_MODEL
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else None
        }
    )
