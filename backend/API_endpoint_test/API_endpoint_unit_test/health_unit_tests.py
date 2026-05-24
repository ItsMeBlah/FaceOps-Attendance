"""
health_unit_tests.py
====================
Unit tests for GET /api/health

This is the first test that runs in CI/CD — it verifies the server
started correctly before any other tests run. If this fails, the
problem is the server environment, not the code.
"""

from __future__ import annotations

import pytest
import requests


ENDPOINT = "/api/health"


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@pytest.mark.order(1)
def test_health_returns_200(base_url, session):
    r = session.get(f"{base_url}{ENDPOINT}", timeout=5)
    assert r.status_code == 200


def test_health_returns_json(base_url, session):
    r = session.get(f"{base_url}{ENDPOINT}", timeout=5)
    assert r.headers["content-type"].startswith("application/json")


def test_health_has_status_key(base_url, session):
    r = session.get(f"{base_url}{ENDPOINT}", timeout=5)
    data = r.json()
    assert "status" in data


def test_health_status_is_ok(base_url, session):
    r = session.get(f"{base_url}{ENDPOINT}", timeout=5)
    data = r.json()
    assert data["status"] == "ok"


def test_health_has_service_key(base_url, session):
    r = session.get(f"{base_url}{ENDPOINT}", timeout=5)
    data = r.json()
    assert "service" in data


def test_health_response_time(base_url, session):
    """
    Health check should be fast — it does no ML inference.
    If this is slow something is wrong with the server environment.
    """
    import time
    start = time.time()
    session.get(f"{base_url}{ENDPOINT}", timeout=5)
    elapsed = time.time() - start
    assert elapsed < 2.0, f"Health check took {elapsed:.2f}s — expected under 2s"
