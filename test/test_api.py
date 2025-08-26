from fastapi.testclient import TestClient
from src.api.fastapi_app import app, get_rag_pipeline
import json

class TestPipeline:
    async def query(self, query: str, config: dict):
        return {
            "answer": "This is a test process",
            "sources": [
                {"content": "This is source content",
                "metadata": {"doc_id": "test_doc", "page": 1}
                }],
                "processing_time": 0.02
        }

    async def stream(self, query: str, config: dict):
        yield {"content": "This is stream test"}
        yield {"done": True}


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
    assert data

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

        for line in response.iter_lines():

            if not line:
                continue
            
            assert line.startswith("data: ")

            try:
                data = json.loads(line[len(b"data: "):])

            except Exception as e:
                assert False, f"Error in json loads: {e}"

            if data.get("done"):
                query_processed = True
                break

        assert query_processed