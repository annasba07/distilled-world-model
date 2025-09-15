import os
import importlib
from fastapi.testclient import TestClient


def get_app_cpu():
    os.environ["LWM_DEVICE"] = "cpu"
    import src.api.server as server
    importlib.reload(server)
    return server.app


def test_replay_determinism_byte_equal():
    app = get_app_cpu()
    with TestClient(app) as client:
        r = client.post("/session/create", json={"prompt": "determinism", "seed": 99})
        assert r.status_code == 200
        sid = r.json()["session_id"]

        # Perform a few steps
        for a in [0, 1, 4, 3, 2, 0]:
            client.post("/session/step", json={"session_id": sid, "action": a})

        snap = client.post("/session/snapshot", json={"session_id": sid}).json()

        # Replay once
        rep1 = client.post("/session/replay", json={
            "prompt": snap["prompt"],
            "seed": snap["seed"],
            "actions": snap["actions"],
            "verify": True
        }).json()
        assert rep1.get("count", 0) >= 1
        ver = rep1.get("verification", {})
        assert ver.get("identical") is True

