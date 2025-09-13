#!/usr/bin/env bash
set -euo pipefail

# Requires GitHub CLI (gh) and authenticated session: gh auth login
# Optionally, set PROJECT_NUM to auto-add items to a specific Projects v2 board.

REPO="${REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
PROJECT_NUM="${PROJECT_NUM:-}"

create_issue() {
  local title="$1"; shift
  local body="$1"; shift
  local labels="$1"; shift
  local issue_url
  issue_url=$(gh issue create --repo "$REPO" --title "$title" --body "$body" --label $labels --json url -q .url)
  echo "Created: $issue_url"
  if [[ -n "$PROJECT_NUM" ]]; then
    gh project item-add $PROJECT_NUM --url "$issue_url"
  fi
}

create_issue "API: /models list/load/status" "Implement endpoints to list model checkpoints, load by id, and report status/progress.\n\nAcceptance:\n- GET /models returns ids + metadata\n- POST /models/load loads by id (async)\n- GET /models/status reports current + progress%\n- Error handling + clear messages" enhancement

create_issue "API: /session/snapshot + /session/replay" "Export/import deterministic sessions (prompt, seed, action timeline, settings).\n\nAcceptance:\n- POST /session/snapshot returns JSON\n- POST /session/replay accepts snapshot and replays deterministically\n- Optional: SSIM check for verification" enhancement

create_issue "API: /settings + auto‑tune + TTL" "Expose resolution/FPS/overlay flags; implement auto‑tune and idle TTL.\n\nAcceptance:\n- GET/POST /settings\n- Auto‑tune policy applied when FPS drops\n- Idle TTL enforced + surfaced in UI" enhancement

create_issue "UI: Model loader + status + sample prompts" "Add loader panel with list, progress, and status; curated sample prompts.\n\nAcceptance:\n- Dropdown of models + Load/Unload\n- Progress indicator + status pill\n- Sample prompt drawer with click‑to‑fill" enhancement

create_issue "UI: Snapshot/Replay + share URLs" "Buttons to export/import snapshot JSON and replay; support prompt/seed URL params.\n\nAcceptance:\n- Export JSON file\n- Import and replay deterministically\n- `?prompt=&seed=` prefill and autogenerate" enhancement

create_issue "UI: Settings panel (FPS, resolution, overlay)" "Panel for key toggles, persistent in localStorage.\n\nAcceptance:\n- FPS target, resolution selector, overlay toggle\n- Save/restore settings across sessions" enhancement

create_issue "QA: Determinism harness + perf matrix" "Test replay determinism (CPU/FP32 exact or SSIM tolerance), record perf at common configs.\n\nAcceptance:\n- Script to replay snapshot and compare\n- Perf table produced for doc" task

create_issue "Docs: Quick‑start + snapshot/replay guide" "README additions and a short how‑to for snapshot/replay and sharing." documentation

echo "All issues created. Configure PROJECT_NUM to auto-add to your project board."

