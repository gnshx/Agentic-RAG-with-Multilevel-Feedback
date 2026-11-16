import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_endpoint():
    with TestClient(app) as c:
        response = c.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "indexed_chunks" in data

def test_query_and_feedback_flow():
    with TestClient(app) as c:
        # 1. Execute query
        q_payload = {"query": "How do feedback loops work in Agentic RAG?", "session_id": "test_api_session"}
        q_res = c.post("/api/query", json=q_payload)
        assert q_res.status_code == 200
        q_data = q_res.json()
        assert q_data["generation"] != ""
        assert q_data["system_log_id"] is not None

        # 2. Submit Level 3 feedback
        log_id = q_data["system_log_id"]
        fb_payload = {
            "inference_id": log_id,
            "rating": 1,
            "comment": "Accurate and verified output",
            "add_to_golden_cache": True
        }
        fb_res = c.post("/api/feedback", json=fb_payload)
        assert fb_res.status_code == 200
        fb_data = fb_res.json()
        assert fb_data["status"] == "success"
        assert fb_data["cached_in_golden"] is True

        # 3. Test Golden Cache short-circuit on next query
        q_res_repeat = c.post("/api/query", json=q_payload)
        assert q_res_repeat.status_code == 200
        q_repeat_data = q_res_repeat.json()
        assert q_repeat_data["cached_answer"] is True

def test_diagnostics_endpoint():
    with TestClient(app) as c:
        res = c.get("/api/diagnostics")
        assert res.status_code == 200
        data = res.json()
        assert "telemetry" in data
        assert "diagnosis" in data

