# FaceOps Attendance

FaceOps Attendance is a full-stack facial recognition attendance system with emotion recognition, anti-spoofing, RTSP camera ingestion, Kafka-based background processing, and MLOps support for dataset/version tracking.

The application is built around a hybrid inference flow:

- FastAPI can run inference immediately for camera frames and API uploads.
- Kafka is used for backend background pipelines, including RTSP frame events, inference result writes, alerts, and cropped face storage.
- The React dashboard does not read Kafka directly. It reads historical attendance, emotion, recognition, and camera data from MongoDB through FastAPI APIs.

## Table of Contents

- [Introduction](#introduction)
- [System Diagram](#system-diagram)
- [Installation](#installation)
- [Running the Stack](#running-the-stack)
- [System Architecture](#system-architecture)
  - [Client Layer](#client-layer)
  - [Backend API Layer](#backend-api-layer)
  - [Kafka Streaming Layer](#kafka-streaming-layer)
  - [AI Inference Layer](#ai-inference-layer)
  - [Async Worker Layer](#async-worker-layer)
  - [Storage Layer](#storage-layer)
  - [Training and MLOps Layer](#training-and-mlops-layer)
- [Repository Structure](#repository-structure)
- [ML Results](#ml-results)
- [Main API Endpoints](#main-api-endpoints)
- [Configuration](#configuration)
- [Development and Validation](#development-and-validation)

## Introduction

FaceOps combines computer vision inference, event streaming, and MLOps tooling into one attendance platform. Users can register faces, run image or camera inference, monitor dashboard statistics, and collect aligned face crops for future model training.

Core capabilities:

- Face detection with crops, bounding boxes, and landmarks.
- Face recognition using Qdrant vector search.
- Emotion classification.
- Anti-spoofing and liveness detection.
- RTSP stream registration and background frame sampling.
- Kafka topics for scalable backend processing.
- MongoDB dashboard data storage.
- MinIO image storage for raw face logs and aligned face datasets.
- Dataset pipeline from MinIO to local data folders and DVC/DagsHub.
- MLflow logging to DagsHub for training metrics, artifacts, and registered models.

## System Diagram

![FaceOps system diagram](assets/FaceOPs%20Diagram.drawio.png)

## Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.12 for local backend development
- Node.js 20 for local frontend development
- Git

Optional:

- NVIDIA Container Toolkit for GPU-backed Triton inference
- DVC and a DagsHub remote for dataset versioning
- DagsHub MLflow credentials for experiment/model logging

### Model Files

Place the ONNX models in `backend/weights`:

```text
backend/weights/
  face_detection.onnx
  anti_spoofing.onnx
  emotion.onnx
  resnet18_face.onnx
```

The Docker stack mounts these model files at runtime. They are not copied into the backend image.

## Running the Stack

Start the full stack:

```bash
docker compose up --build
```

Open the main services:

```text
Frontend:       http://localhost:8080
Backend API:    http://localhost:8000
API docs:       http://localhost:8000/docs
MinIO console:  http://localhost:9001
Qdrant:         http://localhost:6333
Triton HTTP:    http://localhost:8001
```

Default MinIO credentials from `docker-compose.yml`:

```text
Username: faceguard
Password: faceguardsecret
```

Stop the stack:

```bash
docker compose down
```

View logs:

```bash
docker compose logs backend
docker compose logs frontend
docker compose logs kafka
docker compose logs database-writer-worker
docker compose logs face-storage-worker
docker compose logs rtsp-inference-worker
```

Run the GPU Triton stack when NVIDIA Docker is configured:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## System Architecture

### Client Layer

The client layer is the React frontend. It provides the user-facing interface for:

- Dashboard statistics.
- Camera or uploaded-video inference views.
- Face registration.
- Registered face management.
- RTSP stream management.

The frontend calls FastAPI over HTTP. It does not connect to Kafka, MongoDB, MinIO, or Qdrant directly.

### Backend API Layer

The backend API layer is the FastAPI application in `backend/app`.

Responsibilities:

- Expose REST endpoints for dashboard data, face registration, image inference, verification, and RTSP stream control.
- Run immediate inference for direct `/api/pipeline/frame` requests.
- Publish background events to Kafka when asynchronous writes or RTSP inference are needed.
- Read dashboard data from MongoDB and return it to the React frontend.

The backend can run local ONNX inference or connect to Triton depending on configuration.

### Kafka Streaming Layer

Kafka is used only for backend background pipelines. It is not used for login, CRUD, direct dashboard reads, or frontend communication.

Current topics:

```text
camera-frame-events       12 partitions
inference-results         12 partitions
face-storage-requests     12 partitions
```

Topic purpose:

- `camera-frame-events`: RTSP stream readers publish sampled camera frames for inference workers.
- `inference-results`: inference services publish recognition, emotion, and liveness results.
- `face-storage-requests`: inference services publish raw/aligned cropped face images that should be stored.

Consumer groups:

```text
inference-workers
database-writer
alert-worker
face-storage-workers
```

### AI Inference Layer

The AI inference layer runs the computer vision models:

- Face detection service.
- Face recognition service.
- Anti-spoofing service.
- Emotion classification service.

The inference pipeline is:

```text
input image/frame
  -> face detection
  -> face crops and landmarks
  -> anti-spoofing
  -> emotion classification
  -> face recognition with Qdrant
  -> inference result
```

For RTSP streams, workers consume camera frame events, run inference, then publish result and storage events.

### Async Worker Layer

Background workers consume Kafka topics and perform side effects outside the request-response path.

Workers:

- `rtsp-inference-worker`: consumes `camera-frame-events`, runs inference, and publishes result/storage events.
- `database-writer-worker`: consumes `inference-results` and writes attendance/emotion/recognition data for the dashboard.
- `alert-worker`: consumes `inference-results` and can be extended for suspicious or policy-based alerts.
- `face-storage-worker`: consumes `face-storage-requests` and saves cropped face images to MinIO and metadata to MongoDB.

This keeps high-volume camera processing scalable by allowing more worker replicas to be added later.

### Storage Layer

Storage services:

- MongoDB stores attendance sessions, emotion statistics, recognition results, RTSP stream records, and dashboard data.
- MinIO stores image logs and aligned face datasets.
- Qdrant stores registered face embeddings for recognition search.

MinIO buckets:

```text
logs
aligned-images
```

Logged image paths follow the date/user structure:

```text
logs/{date}/{user_name}/images
aligned-images/{date}/{user_name}/images
```

### Training and MLOps Layer

The training and MLOps layer lives under `training_module`.

It includes:

- Dataset extraction from MinIO `aligned-images`.
- Date-range dataset creation.
- Image quality filtering for small or blurry images.
- DVC dataset versioning to DagsHub.
- MLflow experiment logging to DagsHub.
- Training logs, test metrics, artifacts, and registered model logging.
- ONNX conversion and Triton model repository support.

The dataset pipeline creates local datasets named like:

```text
MinIO_Dataset_{start-date}-{end-date}
```

## Repository Structure

```text
.
|-- assets/
|   `-- FaceOPs Diagram.drawio.png
|-- backend/
|   |-- API_endpoint_test/          API endpoint test suite
|   |-- app/
|   |   |-- api/                    FastAPI routes
|   |   |-- configs/                Backend YAML config
|   |   |-- database/               MongoDB helpers
|   |   |-- kafka_messaging/        Kafka topics, schemas, producer, consumer
|   |   |-- logging/                MinIO image logging and Mongo result logging
|   |   |-- schemas/                Pydantic request/response schemas
|   |   |-- services/               Inference, RTSP, vector store, and business services
|   |   |-- utils/                  Preprocessing, alignment, and image helpers
|   |   `-- workers/                Kafka background workers
|   |-- Dockerfile
|   |-- requirements.txt
|   `-- weights/                    Local ONNX model files
|-- frontend/
|   |-- src/                        React application
|   |-- Dockerfile
|   `-- nginx.conf
|-- training_module/
|   |-- dataset_pipeline/           MinIO extraction, transform, and DVC logging
|   |-- mlflow_logging/             DagsHub MLflow logging utilities
|   |-- face_recognition_module/    Face recognition training code
|   |-- emotion_module/             Emotion model training resources
|   |-- anti_spoofing_module_MobileNetV2/
|   `-- Face_detection_module/
|-- triton_models/                  Triton model repository configs
|-- docker-compose.yml
|-- docker-compose.gpu.yml
`-- README.md
```

## ML Results

Current reported model performance:

| Model | Metric |
| --- | ---: |
| Emotion classification | 69.06% accuracy |
| Face recognition | 90.525% accuracy |
| Anti-spoofing | 97.638% accuracy |

## Main API Endpoints

```text
GET  /api/health
POST /api/pipeline/frame
POST /api/detection/detect
POST /api/emotion/
POST /api/anti-spoofing/anti-spoof
POST /api/verification/verify
POST /api/verification/register
POST /api/verification/register-batch
GET  /api/verification/status
```

RTSP endpoints are exposed through the backend RTSP router and are used by the frontend RTSP page to create, list, preview, and delete camera streams.

## Configuration

Main backend config:

```text
backend/app/configs/config.yaml
```

Important Docker environment values:

```text
BACKEND_USE_TRITON=true
TRITON_URL=triton:8000
BACKEND_WEIGHTS_DIR=/app/weights
QDRANT_URL=http://qdrant:6333
MONGODB_URI=mongodb://faceguard:faceguard@mongodb:27017/?authSource=admin
MINIO_ENDPOINT=minio:9000
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
```

RTSP sampling config:

```yaml
rtsp:
  sample_interval_seconds: 1.0
  preview_fps: 8.0
  frame_width: 960
  jpeg_quality: 80
```

## Development and Validation

Run backend locally:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Run frontend locally:

```bash
cd frontend
npm install
npm run dev
```

Run API endpoint tests against a running backend:

```bash
pytest backend/API_endpoint_test -v --base-url http://127.0.0.1:8000
```

The GitHub Actions workflow validates:

- Backend dependency install and Python syntax.
- Required Docker services for API tests.
- FastAPI health check.
- API endpoint tests.
- Frontend production build.
- Docker Compose config and backend/frontend image builds.
