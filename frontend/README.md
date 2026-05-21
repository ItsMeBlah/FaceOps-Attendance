# FaceGuard Frontend

React + Vite frontend for the COS30082 facial recognition attendance system.

## Two pages

The top bar lets you switch between:

1. **§ 01 — The Station** — main attendance UI (camera + live detection + registry + log).
2. **§ 02 — Diagnostics** — per-endpoint test page (run each API individually, see response, latency, status).

The diagnostics page is what you use right now while detection isn't finished — it draws a hardcoded crop box on the camera, takes the crop, and sends it to each endpoint with a "Run test" button.

## Quick start

```bash
cd frontend
cp .env.example .env       # only first time
npm install                 # only first time
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). Make sure the backend is also running at `http://127.0.0.1:8000`.

Switch to the diagnostics page from the top bar, click **Start camera**, then **Run test** on each card.

## What each endpoint card sends

| Endpoint                    | Payload                  | Status now                |
|-----------------------------|--------------------------|---------------------------|
| `/api/pipeline/frame`       | Full frame               | Not finished — placeholder |
| `/api/detection/detect`     | Full frame               | Not finished — placeholder |
| `/api/emotion/`             | Crop (hardcoded box)     | **Wired & testable**      |
| `/api/anti-spoofing/anti-spoof` | Crop (hardcoded box) | **Wired & testable**      |
| `/api/verification/verify`  | Crop (hardcoded box)     | Wired                     |
| `/api/verification/register`| Crop + `person_id`       | Wired                     |

## Configuration

Edit `.env`:

```
VITE_BACKEND_URL=http://127.0.0.1:8000

VITE_ENDPOINT_PIPELINE=/api/pipeline/frame
VITE_ENDPOINT_DETECT=/api/detection/detect
VITE_ENDPOINT_EMOTION=/api/emotion/
VITE_ENDPOINT_SPOOF=/api/anti-spoofing/anti-spoof
VITE_ENDPOINT_VERIFY=/api/verification/verify
VITE_ENDPOINT_REGISTER=/api/verification/register

VITE_FRAME_INTERVAL_MS=2000
VITE_DEMO_PLACEHOLDERS=true
```

The Vite dev server proxies all `/api/*` requests to `VITE_BACKEND_URL`, so the backend URL never appears in client code.

## Backend CORS patch (one-time)

Add to `backend/app/main.py` after `app = FastAPI(...)`:

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Project structure

```
frontend/
├── .env / .env.example     # config
├── vite.config.js          # dev proxy
├── index.html
└── src/
    ├── main.jsx
    ├── App.jsx             # routing + main page state
    ├── components/
    │   ├── TopBar.jsx
    │   ├── CameraView.jsx
    │   ├── DetectionPanel.jsx
    │   ├── RegisteredFaces.jsx
    │   ├── AttendanceLog.jsx
    │   ├── RegisterModal.jsx
    │   ├── Toasts.jsx
    │   ├── Icons.jsx
    │   ├── PlaceholderBadge.jsx
    │   ├── DiagnosticsView.jsx     ← test page
    │   └── EndpointCard.jsx        ← test card
    ├── hooks/
    │   ├── useCamera.js
    │   ├── useFrameAnalysis.js
    │   └── useToasts.js
    ├── services/
    │   └── api.js
    ├── utils/
    │   ├── drawing.js
    │   └── cropBox.js              ← hardcoded crop region
    └── styles/
        ├── tokens.css
        └── global.css
```

## Common issues

| Problem                    | Fix                                                    |
|----------------------------|--------------------------------------------------------|
| Camera access denied       | Allow webcam permission in the browser                 |
| Backend unreachable        | Start backend; check `VITE_BACKEND_URL` in `.env`      |
| White screen on `npm run dev` | Run `npm install` first; check console for errors   |
| 404 on /api/...            | Backend isn't running or proxy target is wrong         |
