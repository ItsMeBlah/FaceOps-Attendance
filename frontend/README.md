# Face Recognition with Emotion and Liveness

React + Vite frontend for the COS30082 facial recognition attendance system.

## Pages

The top bar lets you switch between:

1. **§ 01 — Main Page** — main attendance UI (camera + live detection + registry + log).
2. **§ 02 — Register** — capture or upload exactly five face images and register them together.
3. **§ 03 — Upload Video** — process a local video in real time and download the annotated result.

## Quick start

```bash
cd frontend
cp .env.example .env      
npm install                 
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). Make sure the backend is also running at `http://127.0.0.1:8000`.

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
VITE_ENDPOINT_REGISTER_BATCH=/api/verification/register-batch
VITE_ENDPOINT_VERIFICATION_STATUS=/api/verification/status

VITE_FRAME_INTERVAL_MS=2000
VITE_VIDEO_FRAME_INTERVAL_MS=1000
```

The Vite dev server proxies all `/api/*` requests to `VITE_BACKEND_URL`, so the backend URL never appears in client code.

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
    │   ├── RegistrationView.jsx   ← five-image registration page
    │   ├── Toasts.jsx
    │   ├── Icons.jsx
    │   ├── VideoUploadView.jsx    ← local video analysis + annotated download
    ├── hooks/
    │   ├── useCamera.js
    │   ├── useFrameAnalysis.js
    │   └── useToasts.js
    ├── services/
    │   └── api.js
    ├── utils/
    │   └── drawing.js
    └── styles/
        ├── tokens.css
        └── global.css
```
