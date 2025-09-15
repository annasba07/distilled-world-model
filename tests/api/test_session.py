import os
import importlib
from fastapi.testclient import TestClient


def get_app_cpu():
    os.environ["LWM_DEVICE"] = "cpu"
    import src.api.server as server
    importlib.reload(server)
    return server.app


def test_create_step_snapshot_replay():
    app = get_app_cpu()
    with TestClient(app) as client:
        # Create session
        r = client.post("/session/create", json={"prompt": "2D platformer", "seed": 123})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data and "initial_frame" in data
        sid = data["session_id"]

        # Bad step
        r_bad = client.post("/session/step", json={"session_id": "nope", "action": 0})
        assert r_bad.status_code == 404

        # Good step
        r2 = client.post("/session/step", json={"session_id": sid, "action": 0})
        assert r2.status_code == 200
        data2 = r2.json()
        assert "frame" in data2 and "metrics" in data2

        # Snapshot
        snap = client.post("/session/snapshot", json={"session_id": sid}).json()
        assert snap.get("prompt") == "2D platformer"
        assert snap.get("seed") == 123
        assert isinstance(snap.get("actions"), list)

        # Replay (deterministic CPU path)
        rep = client.post(
            "/session/replay",
            json={
                "prompt": snap["prompt"],
                "seed": snap["seed"],
                "actions": snap["actions"],
                "model": snap.get("model"),
                "settings": snap.get("settings"),
            },
        )
        assert rep.status_code == 200
        frames = rep.json().get("frames", [])
        assert len(frames) >= 1
