# PR: Interactive Prompt‑To‑World MVP

## Summary
Ship an end‑to‑end interactive MVP: users can load a model, enter a prompt, play a 2D world at smooth FPS, and share/replay deterministic sessions from a sleek web UI.

Related milestone: `MILESTONE_INTERACTIVE_MVP.md`

## Changes (Scope)
- API: `/models` list/load/status; `/session/snapshot`, `/session/replay`; `/settings` get/set
- Inference: deterministic seeding; action timeline capture; replay verification (optional SSIM)
- UI: model loader, sample prompts, snapshot/replay, settings; improved metrics and UX
- Performance: auto‑tune resolution/FPS; session TTL; friendly errors/toasts

## Acceptance Criteria
- TTFW: ≤ 3s to first controllable frame from curated prompt
- FPS: ≥ 15 median @ 256×256 on RTX 3060; p95 latency ≤ 120 ms
- Replay determinism: identical/near‑identical frames for same prompt+seed+actions
- Model management: list/load/unload + progress + status
- Sharing: WebM recording, session snapshot export/import, prefilled share URLs

## Tasks (Checklist)
- [ ] Backend: `/models` endpoints (list/load/status)
- [ ] Backend: `/session/snapshot` (export prompt/seed/actions/settings)
- [ ] Backend: `/session/replay` (deterministic playback)
- [ ] Backend: `/settings` get/set; enforce auto‑tune policy; idle TTL
- [ ] Inference: deterministic seeding; ensure action timeline capture
- [ ] UI: Model loader panel + load progress + status pill
- [ ] UI: Sample prompts drawer with curated examples
- [ ] UI: Snapshot export/import; Replay button; support `?prompt=&seed=`
- [ ] UI: Settings panel (FPS target, resolution, overlay toggle); persist locally
- [ ] UI: Improve metrics overlay; timeline polishing; error toasts
- [ ] QA: Replay determinism harness (CPU/FP32 and GPU tolerance)
- [ ] Docs: README quick‑start, settings, and snapshot/replay guide

## How to Test
1. Start API: `python -m src.api.server`; open `/` → redirected to `/demo`
2. Load model via loader; verify status goes from Loading → Ready
3. Choose sample prompt (Platformer) → first frame ≤ 3s
4. Play with WASD/Space; observe FPS/inference metrics & sparkline
5. Export snapshot; replay and compare frames (determinism)
6. Record 10s; download WebM; verify playback
7. Change FPS/resolution; verify auto‑tune and total experience smoothness

## Screenshots / Clips
> Attach UI screenshots and a short clip demonstrating generate → play → snapshot → replay → record

## Notes
- Fallback behavior: if performance drops, UI prompts to lower resolution; TTL countdown for idle sessions
- Future follow‑up: multiverse generate and guided onboarding

