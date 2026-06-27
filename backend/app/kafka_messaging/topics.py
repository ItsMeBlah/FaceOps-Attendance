# Kafka is used only for backend background writes.
# React, login, CRUD, and FastAPI dashboard reads must not use Kafka directly.
INFERENCE_RESULTS_TOPIC = "inference-results"
FACE_STORAGE_REQUESTS_TOPIC = "face-storage-requests"

ALL_TOPICS = (
    INFERENCE_RESULTS_TOPIC,
    FACE_STORAGE_REQUESTS_TOPIC,
)

DATABASE_WRITER_GROUP = "database-writer"
ALERT_WORKER_GROUP = "alert-worker"
FACE_STORAGE_WORKER_GROUP = "face-storage-workers"
