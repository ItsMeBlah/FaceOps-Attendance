# Facial Recognition with Emotion and Liveness

Full-stack face analysis application with:

- face detection
- anti-spoofing / liveness
- emotion recognition
- face verification
- Qdrant vector search
- Triton Inference Server support

## Project Structure

```text
backend/
  app/
    api/          FastAPI routes
    schemas/      API response schemas
    services/     inference and vector database services
    utils/        image preprocessing and face alignment
    configs/      YAML config
  weights/        ONNX model files, not committed

frontend/
  src/            React app
  nginx.conf      Docker Nginx config and backend proxy

triton_models/
  */config.pbtxt  Triton model configs
  start-triton.sh Builds Triton model repository from backend/weights

docker-compose.yml
docker-compose.gpu.yml
```

## Required Model Files

Put ONNX files in `backend/weights`:

```text
backend/weights/
  face_detection.onnx
  anti_spoofing.onnx
  emotion.onnx
  resnet18_face.onnx
```

The model files are mounted into Docker at runtime. They are not copied into images.

## Run with Docker

From the project root:

```bash
docker compose up --build
```

Open:

```text
Frontend:     http://localhost:8080
API docs:     http://localhost:8080/docs
Backend docs: http://localhost:8000/docs
Qdrant:       http://localhost:6333
Triton HTTP:  http://localhost:8001
```

Stop services:

```bash
docker compose down
```

View logs:

```bash
docker compose logs backend
docker compose logs frontend
docker compose logs triton
docker compose logs qdrant
```

## GPU Triton

Default Docker Compose runs Triton without requiring GPU access. This avoids startup failure on machines where NVIDIA Docker is not configured.

To run Triton with GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

If you see:

```text
failed to discover GPU vendor from CDI
```

or:

```text
NVIDIA Driver was not detected
```

then Docker cannot access the GPU yet. Test with:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that fails, install/configure NVIDIA Container Toolkit, then retry the GPU Compose command.

## Main API Endpoints

```text
POST /api/pipeline/frame
POST /api/detection/detect
POST /api/emotion/
POST /api/anti-spoofing/anti-spoof
POST /api/verification/verify
POST /api/verification/register
POST /api/verification/register-batch
GET  /api/verification/status
```

The frontend proxies `/api/*`, `/docs`, `/openapi.json`, and `/redoc` to the backend through Nginx.

## Configuration

Main config:

```text
backend/app/configs/config.yaml
```

Important runtime env values are set in `docker-compose.yml`:

```text
BACKEND_USE_TRITON=true
TRITON_URL=triton:8000
QDRANT_URL=http://qdrant:6333
BACKEND_WEIGHTS_DIR=/app/weights
BACKEND_CAPTURES_DIR=/app/captures
```

## Local Development

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

For local frontend development, use `frontend/.env` to point Vite proxy to the backend:

```text
VITE_BACKEND_URL=http://127.0.0.1:8000
```