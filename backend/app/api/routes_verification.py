from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from uuid6 import uuid7

from app.schemas.verification_schema import RecognitionResponse, RecognitionResult, RegisterResponse
from app.services.inference_service import InferenceService
from app.services.verification_service import VerificationService
from app.utils.preprocess import load_image_from_bytes


router = APIRouter()
verification_service = VerificationService()
inference_service = InferenceService()


async def _load_upload_image(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    try:
        contents = await file.read()
        image = load_image_from_bytes(contents)
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image size")
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")


@router.get("/status")
async def verification_status() -> dict:
    store = verification_service.vector_store
    try:
        store._ensure_collection()
        points_count = store.client.count(
            collection_name=store.collection_name,
            exact=True,
        ).count
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}")

    return {
        "qdrant": "ok",
        "collection": store.collection_name,
        "vector_size": store.vector_size,
        "top_k": store.top_k,
        "points": points_count,
        "max_registration_images": verification_service.max_registration_images,
    }


@router.post("/verify", response_model=RecognitionResponse)
async def verify_face(file: UploadFile = File(...)) -> RecognitionResponse:
    image = await _load_upload_image(file)

    try:
        result = verification_service.verify(image)[0]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return RecognitionResponse(recognition=RecognitionResult(**result))


@router.post("/register", response_model=RegisterResponse)
async def register_face(
    person_name: str = Form(...),
    file: UploadFile = File(...),
) -> RegisterResponse:
    image = await _load_upload_image(file)
    person_name = person_name.strip()
    if not person_name:
        raise HTTPException(status_code=400, detail="Person name is required.")
    person_id = str(uuid7())

    try:
        result = inference_service.register_inference(
            [image],
            person_id,
            person_name=person_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return RegisterResponse(
        person_id=result["person_id"],
        status=result["status"],
        message=f"Stored {result['image_count']} face embedding(s) in Qdrant.",
    )


@router.post("/register-batch", response_model=RegisterResponse)
async def register_faces(
    person_name: str = Form(...),
    files: list[UploadFile] = File(...),
) -> RegisterResponse:
    if len(files) > verification_service.max_registration_images:
        raise HTTPException(
            status_code=400,
            detail=f"Upload at most {verification_service.max_registration_images} images.",
        )
    person_name = person_name.strip()
    if not person_name:
        raise HTTPException(status_code=400, detail="Person name is required.")
    person_id = str(uuid7())

    images = []
    for file in files:
        images.append(await _load_upload_image(file))

    try:
        result = inference_service.register_inference(
            images,
            person_id,
            person_name=person_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return RegisterResponse(
        person_id=result["person_id"],
        status=result["status"],
        message=f"Stored {result['image_count']} face embedding(s) in Qdrant.",
    )
