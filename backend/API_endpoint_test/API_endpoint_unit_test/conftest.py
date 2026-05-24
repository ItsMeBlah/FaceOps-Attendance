"""
conftest.py
===========
Shared pytest fixtures for both unit and integration tests.
Lives in API_endpoint_test/ — shared across:
    API_endpoint_unit_test/
    API_endpoint_integration_test/

pytest automatically loads this file before running any tests —
no import needed in the test files themselves.

Fixtures defined here:
    base_url          — server base URL, overridable via --base-url CLI option
    session           — shared requests.Session reused across all tests
    sample_image      — raw bytes of a small valid JPEG for structural tests
    blank_image       — raw bytes of a blank white JPEG for edge case tests
    bad_file          — raw bytes of a plain text file for 400 error tests
    face_image_bytes  — raw bytes of a real face image from test_assets/
    server_available  — checks server is up before running any tests
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import requests
from PIL import Image


# ------------------------------------------------------------------
# Test assets path
# __file__ = API_endpoint_test/conftest.py
# test_assets/ is a sibling of the two test folders, inside API_endpoint_test/
# ------------------------------------------------------------------

TEST_ASSETS_DIR   = Path(__file__).parent / "test_assets"
DEFAULT_FACE_IMAGE = TEST_ASSETS_DIR / "test_face.jpg"


# ------------------------------------------------------------------
# CLI options
# ------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        action="store",
        default="http://127.0.0.1:8000",
        help="Base URL of the running FaceGuard API server",
    )
    parser.addoption(
        "--image",
        action="store",
        default=str(DEFAULT_FACE_IMAGE),
        help="Path to a real face image for tests that require detection",
    )


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_url(request) -> str:
    """
    Base URL of the API server.
    Session-scoped — created once for the entire test run.
    Override with: pytest --base-url http://other-host:8000
    """
    return request.config.getoption("--base-url").rstrip("/")


@pytest.fixture(scope="session")
def session() -> requests.Session:
    """
    Shared requests.Session for all tests.
    Session-scoped — one HTTP session reused across all test files.
    More efficient than opening a new connection per test.
    """
    s = requests.Session()
    yield s
    s.close()


@pytest.fixture(scope="session")
def sample_image() -> bytes:
    """
    A minimal valid JPEG image for structural/format tests.
    Programmatically generated — no file dependency, works in CI.
    Small (100x100) to keep tests fast.

    Use this when the test only checks response structure —
    keys, types, status codes — and doesn't need a real face.
    """
    img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="session")
def blank_image() -> bytes:
    """
    A plain white JPEG — used for edge case tests.
    Endpoints should return 200 with empty/default results, not crash.
    Detection should return faces: [].
    Emotion and anti-spoofing should still return a prediction, not 500.
    """
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="session")
def bad_file() -> bytes:
    """
    Plain text bytes — used to test that endpoints return 400
    when a non-image file is uploaded.
    """
    return b"this is not an image file"


@pytest.fixture(scope="session")
def face_image_bytes(request) -> bytes:
    """
    Raw bytes of a real face image from test_assets/.
    Used for tests that require actual face detection to work.

    Default path: API_endpoint_test/test_assets/test_face.jpg
    Override with: pytest --image path/to/other/image.jpg

    Skips gracefully if the file is not found rather than crashing —
    safe for CI environments where the image hasn't been provided yet.
    """
    path = Path(request.config.getoption("--image"))
    if not path.is_file():
        pytest.skip(
            f"No face image found at {path} — "
            f"add a face image to test_assets/test_face.jpg "
            f"or pass --image path/to/image.jpg"
        )
    return path.read_bytes()


@pytest.fixture(scope="session")
def server_available(base_url, session) -> bool:
    """
    Checks the server is reachable before running any tests.
    Session-scoped — checked once for the entire test run.
    """
    try:
        r = session.get(f"{base_url}/api/health", timeout=5)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


@pytest.fixture(autouse=True)
def skip_if_server_down(server_available):
    """
    Automatically applied to every single test — no need to add it manually.
    Skips with a clear message if the server is not running,
    rather than failing with a confusing ConnectionError.

    In CI this immediately tells you the problem is server startup,
    not the test code itself.
    """
    if not server_available:
        pytest.skip("Server not reachable — start uvicorn before running tests")
