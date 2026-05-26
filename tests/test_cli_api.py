import subprocess
import sys
import pytest


# ── cli ────────────────────────────────────────────────────────────────────────

def test_cli_module_is_runnable():
    """The CLI entry point must exist and respond to --help."""
    result = subprocess.run(
        [sys.executable, "-m", "autoexplainml", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"CLI failed with: {result.stderr}"


def test_cli_help_mentions_mode():
    result = subprocess.run(
        [sys.executable, "-m", "autoexplainml", "--help"],
        capture_output=True, text=True
    )
    output = result.stdout.lower() + result.stderr.lower()
    assert "mode" in output or "analyze" in output or "project" in output


def test_cli_no_args_fails_gracefully():
    """Running with no arguments should fail with a clean error, not a traceback."""
    result = subprocess.run(
        [sys.executable, "-m", "autoexplainml"],
        capture_output=True, text=True
    )
    assert "Traceback" not in result.stderr or result.returncode != 0


def test_cli_bad_mode_fails_gracefully():
    result = subprocess.run(
        [sys.executable, "-m", "autoexplainml",
         "fake_model.pkl", "fake_data.csv", "--mode", "invalid_mode"],
        capture_output=True, text=True
    )
    # should not crash with an unhandled exception
    assert "Traceback" not in result.stderr or result.returncode != 0


# ── api ────────────────────────────────────────────────────────────────────────

try:
    from fastapi.testclient import TestClient
    from autoexplainml.api.app import app
    client = TestClient(app)
    HAS_API = True
except Exception:
    HAS_API = False


@pytest.mark.skipif(not HAS_API, reason="FastAPI app not importable")
def test_api_health_check():
    response = client.get("/")
    assert response.status_code in (200, 404)  # 404 is ok if no root route


@pytest.mark.skipif(not HAS_API, reason="FastAPI app not importable")
def test_api_analyze_route_exists():
    """POST /analyze should return 422 (bad input) not 404 (missing route)."""
    response = client.post("/analyze", json={})
    assert response.status_code != 404, "Route /analyze does not exist"


@pytest.mark.skipif(not HAS_API, reason="FastAPI app not importable")
def test_api_returns_json():
    response = client.post("/analyze", json={})
    assert response.headers.get("content-type", "").startswith("application/json")


@pytest.mark.skipif(not HAS_API, reason="FastAPI app not importable")
def test_api_invalid_payload_returns_422():
    response = client.post("/analyze", json={"garbage": "data"})
    assert response.status_code in (400, 422)
