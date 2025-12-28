# Repository Guidelines

## Project Structure & Modules
- `tennis_video/`: Core pose/detection wrappers (`pose_backend_mmpose.py`), video analysis helpers (`analyze_video.py`), and configuration (`config.py`).
- `tennis_video/impact_detection.py`: Estimates `t_impact` from wrist-speed peaks if you don’t want to supply it manually.
- `tools/`: Operator scripts for batch/template generation (e.g., `batch_build_templates.py`, `build_expert_template.py`).
- `configs/` and `checkpoints/`: Local model configs/weights for MMDetection/MMPose; synced to `tennis_video/config.py`.
- `data/`: Input manifests and source videos; outputs (e.g., `expert_templates/`) are written alongside.

## Build, Run, and Dev Commands
- Install env: `conda activate tcvision` (or matching env) with mmdet/mmpose deps preinstalled.
- Generate templates: `python tools/batch_build_templates.py --csv data/indoor_field_manifest_est.csv --output-root expert_templates`
- Single template: `python tools/build_expert_template.py --video PATH --t-impact 12.3 --stroke-type serve`
- Quick import check (no run): `python - <<'PY'\nimport tools.batch_build_templates\nprint('imports ok')\nPY`

## Coding Style & Naming
- Python, 4-space indent, type hints preferred; keep functions small and reusable.
- Config paths in `tennis_video/config.py`; prefer local files, fall back to URLs only if present.
- Add brief, high-value comments for non-obvious logic; avoid noise.
- Paths are repository-relative; prefer `Path` over string concatenation.

## Testing Guidelines
- No formal test suite today; add lightweight checks where practical (e.g., sanity-running a short clip).
- For new logic, add minimal reproducible scripts under `tools/` or docstring examples.
- Keep manual runs deterministic: fix seeds and use small sample videos.

## Commit & PR Practices
- Commit messages: short imperative summary (e.g., “Fix pose backend fallback”).
- PRs: describe intent, list key changes, mention tested commands, and note model/config updates or new assets.
- Include before/after evidence for behavioral changes (logs, small gifs, or CLI output excerpts).

## Security & Configuration Tips
- Avoid embedding secrets; configs should point to local files in `configs/` and `checkpoints/`.
- When GPUs are unavailable, set `DEVICE = "cpu"` in `tennis_video/config.py`; models auto-fallback if CUDA fails.
- Keep downloaded weights/configs under version control only if required; large artifacts should remain cached locally.***
