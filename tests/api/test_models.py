import os
import importlib
from fastapi.testclient import TestClient


def get_app_cpu():
    os.environ["LWM_DEVICE"] = "cpu"
    # Reload module to pick up env
    import src.api.server as server
    importlib.reload(server)
    return server.app


def test_models_list_and_status():
    app = get_app_cpu()
    with TestClient(app) as client:
        r = client.get("/models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "current_model" in data
        
        r2 = client.get("/models/status")
        assert r2.status_code == 200
        s = r2.json()
        assert "current_model" in s
        assert "ready" in s


def test_models_load_404():
    app = get_app_cpu()
    with TestClient(app) as client:
        r = client.post("/models/load", json={"id": "not_a_real_model.pt"})
        assert r.status_code == 404
