import os
from fastapi.testclient import TestClient
from src.api.fastapi_app import app, get_rag_pipeline
import json
import pytest

@pytest.mark.skipif(not os.path.exists("src/inference_repo/fine_tuned_checkpoint_9.pth"),
                    reason="Checkpoint missing")
class TestPipeline:
    async def query_test(self, query: str, config: dict):
        return {"result": [{"content": "This is a test process"}]}


def override_testpipeline():
    return TestPipeline()

app.dependency_overrides[get_rag_pipeline] = override_testpipeline

client = TestClient(app=app)

def test_query():

    payload = {
        "query": "what is fever?",
        "max_new_tokens": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40
    }

    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

def test_stream():

    payload = {
        "query": "what is fever?",
        "max_new_tokens": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40
    }

    with client.stream(method="POST", url="/stream", json=payload) as response:
        assert response.status_code == 200

        assert response.headers.get("content-type").startswith("text/event-stream")

        query_processed = False

        for line in response.aiter_lines():

            if not line:
                continue
            
            assert line.startswith("data: ")

            try:
                data = json.loads(line[2:])

            except Exception as e:
                assert False, f"Error in json loads: {e}"

            if data.get("done"):
                query_processed = True
                break

        assert query_processed