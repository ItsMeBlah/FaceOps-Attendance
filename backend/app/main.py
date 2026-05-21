from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from app.api.routes_anti_spoofing import router as anti_spoofing_router
from app.api.routes_detection import router as detection_router
from app.api.routes_emotion import router as emotion_router
from app.api.routes_pipeline import router as pipeline_router
from app.api.routes_verification import router as verification_router

app = FastAPI(title="FaceGuard API", version="0.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5501",
        "http://127.0.0.1:5501",
        "null",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "FaceGuard API"}

app.include_router(detection_router, prefix="/api/detection", tags=["detection"])
app.include_router(emotion_router, prefix="/api/emotion", tags=["emotion"])
app.include_router(anti_spoofing_router, prefix="/api/anti-spoofing", tags=["anti-spoofing"])
app.include_router(verification_router, prefix="/api/verification", tags=["verification"])
app.include_router(pipeline_router, prefix="/api/pipeline", tags=["pipeline"])

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)