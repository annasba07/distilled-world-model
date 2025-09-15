#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Compile sanity"
python -m compileall -q src demo

echo "[2/4] API + determinism tests"
pytest -q tests -o addopts=''

echo "[3/4] Health check"
python - << 'PY'
import requests, time
import subprocess, sys, os
from threading import Thread

proc = subprocess.Popen([sys.executable, '-m', 'src.api.server'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(2)
ok = False
for _ in range(20):
    try:
        r = requests.get('http://localhost:8000/health', timeout=1)
        if r.ok and r.json().get('engine_loaded'):
            ok = True
            break
    except Exception:
        time.sleep(0.5)
if not ok:
    proc.kill(); sys.exit(1)
print('Server healthy')
proc.kill()
PY

echo "[4/4] Done"

