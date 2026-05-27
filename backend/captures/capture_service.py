import json
import time
from datetime import datetime

from app.config import settings

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

min_interval_secs = 10
only_if_faces = True

last_save_at = 0.0

def save(image_bytes, payload):
    global last_save_at

    if not settings.captures_enabled:
        return

    if only_if_faces and not payload.get("faces"):
        return
    
    current_time = time.monotonic()

    if min_interval_secs and current_time - last_save_at < min_interval_secs:
        return

    last_save_at = current_time

    output_dir = settings.captures_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    image_path = output_dir / f"{timestamp}.jpg"
    json_path = output_dir / f"{timestamp}.json"

    image_path.write_bytes(image_bytes)

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    