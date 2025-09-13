# Milestone: Interactive Prompt‑To‑World MVP

Goal: Ship a delightful, reproducible interactive demo where users load a model, enter a prompt, play a 2D world at smooth FPS, and share/replay sessions — all on consumer hardware.

## Outcome (What “Done” Means)
- Non‑experts can run the server, open the web UI, generate a world from a text prompt, control it with WASD/Space or on‑screen controls, and reliably share & replay a session.
- One command starts everything; first‑try success from curated prompts.

## Scope
- Model Loader: List and load checkpoints in the UI with progress and clear status.
- Sample Prompts: Curated set (e.g., Platformer, Top‑Down, Physics) for first‑try success.
- Generate & Control: Text prompt → initial frame in ≤3s; WASD + Space + D‑Pad controls.
- Snapshot & Replay: Save session (prompt, seed, action timeline), replay deterministically.
- Recording: One‑click WebM recording of the canvas.
- Performance Overlay: Live FPS/inference time, sparkline trend, recent‑frame timeline.
- Health & UX: Server/engine readiness indicator, friendly errors, session ID copy, theme toggle.

## Acceptance Criteria
- Prompt → Playable
  - Generate from curated prompts and obtain a controllable world ≤3s on a mid‑range GPU (e.g., RTX 3060).
- FPS Target
  - ≥15 FPS median at 256×256; p95 latency ≤120 ms; degrades gracefully (auto‑reduce resolution/step rate if needed).
- Deterministic Replay
  - Same prompt+seed+actions reproduces frames within a perceptual tolerance (e.g., SSIM ≥0.98 or exact on CPU/FP32).
- Model Management
  - UI lists checkpoints; load/unload succeeds; status endpoint reflects progress/errors.
- Sharing
  - Record WebM; export/import session snapshot JSON; “Share URL” pre‑populates prompt/seed.
- Stability
  - No dead‑ends (random noise) when a checkpoint is loaded; friendly toasts for errors; idle sessions TTL applied.

## API Additions
- `GET /models` → list available checkpoints and metadata (id, name, size, dtype).
- `POST /models/load` → `{ id: string }` to load a checkpoint asynchronously.
- `GET /models/status` → load progress and current model.
- `POST /session/snapshot` → returns `{ prompt, seed, actions, timestamps, settings }`.
- `POST /session/replay` → accepts snapshot and replays deterministically.
- `POST /settings` / `GET /settings` → resolution, FPS target, overlay flags.

## UI Additions
- Loader Panel: Dropdown of checkpoints, Load/Unload with progress.
- Sample Prompts Drawer: Click‑to‑fill prompts with a short “What you should see”.
- Snapshot/Replay: Save JSON; replay locally or via URL parameter.
- Settings Panel: FPS target, resolution, overlays; persisted in localStorage.
- Enhanced Metrics: Overlay on canvas; sparkline; timeline; session TTL banner.

## Demo Flow (E2E)
1. Start server: `python -m src.api.server`, open `http://localhost:8000/` (redirects to `/demo`).
2. Load model: choose “Tiny” (or any) checkpoint; progress shown.
3. Generate: pick “2D Platformer” → first frame in ≤3s.
4. Play: WASD/Space or D‑Pad; FPS ~15–30 on mid‑range GPU.
5. Snapshot: save JSON; click Replay → identical/near‑identical frames.
6. Record: click Record, play, Stop → WebM downloads.

## Quality Bar
- Performance: 256×256; 15–30 FPS on RTX 3060; under 8 GB VRAM.
- Robustness: Validations, rate limits, idle TTL; clear UI messages for errors.
- Reproducibility: Deterministic seeds; replay within tolerance.
- Usability: No surprises; curated prompts; clear feedback and affordances.

## Why This Milestone
- Delivers an immediately delightful interactive experience.
- Enables reproducible research and debugging via snapshot/replay.
- Establishes infra/UI that scales to improved models (multiverse, advanced dynamics) without re‑architecture.

## Follow‑Ups (Post‑MVP)
- Multiverse Generate: Produce 3–6 candidate initial states; pick the best.
- Guided Onboarding: “Try These” flows and tooltips for first‑time users.
- Settings Profiles: Quality/Performance presets; auto‑tune based on hardware.
- Server‑Side Recording: Optional MP4 or longer recordings without client CPU cost.

