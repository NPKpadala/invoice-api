from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import anthropic
import base64
import io
import os
import re

app = FastAPI(title="Indian GST Invoice Extractor", version="3.0")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Extract all text from this Indian GST invoice. Return the raw text exactly as seen."
                    }
                ]
            }]
        )

        text = message.content[0].text
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
    return {"status": "healthy", "version": "3.0"}
```

---

**Add API key in Render:**
```
render.com → invoice-api → Environment
Add:
Key: ANTHROPIC_API_KEY
Value: your-api-key-here
Save
```

---

**Get your Claude API key:**
```
console.anthropic.com
→ API Keys
→ Create Key
→ Copy
