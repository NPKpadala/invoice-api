# Automated Invoice Data Extraction API

**Production-grade OCR ingestion service** — upload multi-page PDF/image invoices, receive clean structured JSON. Built with Python + FastAPI, hardened with per-client rate limiting, and architected to run behind an Nginx reverse proxy under systemd process supervision.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Nginx](https://img.shields.io/badge/Nginx-009639?style=flat-square&logo=nginx&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-4A90D9?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)

---

## 1. System Architecture

Request lifecycle, edge to extraction:

```
┌────────────────┐      ┌───────────────────────────┐      ┌──────────────────────────┐
│ Client Request │ ───> │ Nginx Reverse Proxy       │ ───> │ Uvicorn ASGI Server      │
│ (PDF / image)  │      │ (Port 80/443, TLS, 50M    │      │ (127.0.0.1:8000)         │
└────────────────┘      │  body limit, long timeouts)│      └────────────┬─────────────┘
                        └───────────────────────────┘                   │
                                                                        ▼
                        ┌───────────────────────────┐      ┌──────────────────────────┐
                        │ OCR Processing Engine     │ <─── │ FastAPI App Layer        │
                        │ OpenCV + Pillow pre-      │      │ (validation, CORS,       │
                        │ processing → Tesseract →  │      │  slowapi rate limiting)  │
                        │ field parsing → JSON      │      └──────────────────────────┘
                        └───────────────────────────┘
```

**Pipeline detail:** uploads are validated at the app layer, pre-processed with OpenCV/Pillow (deskew, contrast enhancement, noise filtering) to maximize OCR accuracy on real-world scans, passed through Tesseract, then parsed into structured fields (vendor, dates, line totals) and returned as JSON.

---

## 2. Infrastructure & Dispatch Specifications

| Component | Specification |
|:---|:---|
| **Web Server** | Nginx — reverse proxy with custom timeout buffers sized for heavy OCR payloads |
| **Application Server** | Uvicorn ASGI worker processes serving the FastAPI application |
| **OCR Engine** | Tesseract OCR with OpenCV + Pillow image pre-processing pipeline |
| **Rate Limiting** | slowapi (per-client-IP) enforced at the application layer |
| **System Requirements** | Linux (Ubuntu / CentOS), optimized for file I/O; `tesseract-ocr` system binary |
| **Process Supervision** | systemd unit — automatic restart on failure, starts on boot |
| **Alt. Deployment** | Dockerfile + `render.yaml` included for container/PaaS deployment |

---

## 3. Nginx Reverse Proxy Configuration

`/etc/nginx/sites-available/invoice-extractor`

```nginx
server {
    listen 80;
    server_name invoices.example.com;

    # Large multi-page PDF/image invoice uploads
    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Heavy text-extraction jobs — prevent 504 Gateway Timeouts
        proxy_connect_timeout 75s;
        proxy_send_timeout    300s;
        proxy_read_timeout    300s;

        # Buffer large OCR responses
        proxy_buffering on;
        proxy_buffers 16 16k;
        proxy_buffer_size 32k;
    }
}
```

> **TLS:** terminate HTTPS on port 443 with your certificate of choice (e.g. `certbot --nginx`) — the proxy block above is unchanged.

---

## 4. Production Deployment & Automation Sequence

### Step 1 — Environment Setup

```bash
# Clone and enter the repository
git clone https://github.com/NPKpadala/invoice-api.git
cd invoice-api

# Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# System dependencies — OCR binaries and image libraries
# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y tesseract-ocr libgl1
# CentOS/RHEL:
sudo dnf install -y tesseract mesa-libGL
```

### Step 2 — Configuration Management

```bash
# Application configuration lives in .env (never committed)
cat > .env <<'EOF'
APP_ENV=production
LOG_LEVEL=info
UPLOAD_DIR=/var/lib/invoice-api/uploads
API_KEY=<generate-a-strong-key>
EOF

sudo mkdir -p /var/lib/invoice-api/uploads
```

### Step 3 — Nginx Linkage

```bash
# Enable the server block
sudo ln -s /etc/nginx/sites-available/invoice-extractor \
           /etc/nginx/sites-enabled/invoice-extractor

# Validate configuration syntax, then reload without dropping connections
sudo nginx -t
sudo systemctl reload nginx
```

### Step 4 — Systemd Service Control

`/etc/systemd/system/invoice-api.service`

```ini
[Unit]
Description=Automated Invoice Data Extraction API (FastAPI/Uvicorn)
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/invoice-api
EnvironmentFile=/opt/invoice-api/.env
ExecStart=/opt/invoice-api/.venv/bin/uvicorn main:app \
          --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Register, start on boot, and launch now
sudo systemctl daemon-reload
sudo systemctl enable --now invoice-api
```

---

## 5. Monitoring & Log Management

| Task | Command |
|:---|:---|
| Application state | `systemctl status invoice-api` |
| Live application logs (follow) | `journalctl -u invoice-api -f` |
| Logs since last boot | `journalctl -u invoice-api -b` |
| Ingress traffic audit | `sudo tail -f /var/log/nginx/access.log` |
| Proxy/edge errors (504s, body-size rejections) | `sudo tail -f /var/log/nginx/error.log` |
| Restart after config change | `sudo systemctl restart invoice-api` |

**Operational notes**

- A `504` in the Nginx error log during large jobs means `proxy_read_timeout` needs widening — the app is still processing.
- A `413 Request Entity Too Large` means the upload exceeded `client_max_body_size`.
- Rate-limited clients receive `429 Too Many Requests` from the application layer (slowapi) — visible in `journalctl`, not Nginx.

---

<div align="center">
Built and operated by <a href="https://npkpadala.com">Praveen Kumar Padala</a> — part of my production systems portfolio.
</div>
