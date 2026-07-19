# Invoice API

OCR-powered invoice data extraction as a REST API: upload an invoice image, get structured JSON back.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-4A90D9)

## Highlights

- **FastAPI** service with CORS support and per-client **rate limiting** (slowapi).
- **Image pre-processing pipeline** — OpenCV + Pillow enhancement/filtering before OCR for better recognition on real-world scans.
- **Tesseract OCR** with field parsing to pull totals, dates, and vendor details out of free-form invoice layouts.
- **Deploy-ready** — Dockerfile, Procfile and `render.yaml` included.

## Run it

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

---

Part of my portfolio — more at [npkpadala.com](https://npkpadala.com).
