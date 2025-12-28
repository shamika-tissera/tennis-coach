# Tennis Coach Vision Web UI

## What it does
- Interactive 3D playback of a user stroke (stick-figure skeleton by default; optional SMPL-style mesh if generated/attached)
- Side‑by‑side or overlay comparison against the matched expert template
- Coach notes (existing `generate_feedback`)
- “Ask the Coach” chat panel (OpenAI‑compatible API)

## Run the server
From the repo root:

```bash
python tools/run_webapp.py
```

Then open `http://127.0.0.1:8000`.

## Create a session
### Option A: From the UI
Click **New Session**, enter `video_path` and (optionally) `t_impact`, then **Create**.

### Option B: From the CLI
```bash
python tools/build_web_session.py --video sample_video.mp4 --stroke-type straight_shot
```

Sessions are written to `data/web_sessions/<session_id>/session.json`.

### Multi-shot clips
If a single video contains multiple strokes, enable **Detect multiple shots (auto)** in the UI (or provide multiple `t_impact` values by creating multiple sessions manually).

## LLM configuration (chat)
Set these env vars before starting the server:

```bash
export TCVISION_LLM_API_KEY="..."
export TCVISION_LLM_BASE_URL="https://integrate.api.nvidia.com/v1"   # optional
export TCVISION_LLM_MODEL="openai/gpt-oss-20b"                       # optional
```

## Notes
- Session creation requires working MMDetection/MMCV + MMPose installs (the UI can still *view* existing sessions without them).
- The 3D viewer uses `three.js` (loads from `/static/vendor/` if present, otherwise from a CDN).
- If a session has no attached mesh, the viewer falls back to rendering `user.joints3d_resampled` / `expert.joints3d_resampled` as a stick-figure skeleton.

## SMPL / MMHuman3D mesh (body surface)
To render a full SMPL-style mesh in the UI, the session needs:
- `mesh.faces` (F×3 triangle indices)
- `user.mesh_vertices_path` (float32 little-endian, frames×verts×3)
- optionally `expert.mesh_vertices_path`

### Automatic mesh generation (recommended)
The UI can generate and attach meshes automatically in two ways:

#### Option A (built-in, most reliable)
This uses the session’s existing `joints3d_resampled` to skin a dense SMPL mannequin (no MMCV/MMHuman3D/MediaPipe required).

Set:
```bash
export TCVISION_SMPL_MODEL="/path/to/SMPL_NEUTRAL.pkl" # enables face fallback + part colors
```

Then in the UI, select a session and click **Generate 3D Mesh**.

#### Option B (external exporter, e.g. MMHuman3D)
Set a generator command on the server:
```bash
export TCVISION_MESH_GENERATOR_CMD="python /path/to/your_mmhuman3d_export.py --video {video} --out {out_npz}"
export TCVISION_MESH_CLIP_TO_WINDOW=1                 # optional (default 1)
export TCVISION_MESH_GENERATOR_TIMEOUT_S=3600         # optional
export TCVISION_SMPL_MODEL="/path/to/SMPL_NEUTRAL.pkl" # optional (enables part colors)
```

Notes:
- If you use `python ...`, point it at a `.py` file (not a directory), or use `python -m some.module`.
- Supported placeholders: `{video}` `{out_npz}` `{start}` `{end}`.

The external command must write an `.npz` containing:
- `vertices` (T, V, 3) float
- `faces` (F, 3) int (optional if `TCVISION_SMPL_MODEL` is set)

If you already have mesh outputs (e.g., from MMHuman3D), you can attach them to a session:
```bash
python tools/attach_mesh_to_session.py --session-id <SESSION_ID> --user-mesh-npz user_mesh.npz --expert-mesh-npz expert_mesh.npz
```

The `.npz` is expected to contain:
- `vertices` as `(T, V, 3)` (or `verts` / `pred_vertices`)
- `faces` as `(F, 3)` (in at least one of the NPZ files)

### Body-part colors (DensePose-style regions on the mesh)
If you also provide per-vertex part labels, the viewer can render a filled mesh where each body region is colored.

Option A (SMPL model → labels):
```bash
python tools/export_smpl_parts.py --smpl-model /path/to/SMPL_NEUTRAL.pkl --out-labels smpl_vertex_labels.npy --out-meta smpl_parts_meta.json
python tools/attach_mesh_to_session.py --session-id <SESSION_ID> --user-mesh-npz user_mesh.npz --expert-mesh-npz expert_mesh.npz --vertex-labels-npy smpl_vertex_labels.npy --parts-meta-json smpl_parts_meta.json
```

Option B (you already have labels):
```bash
python tools/attach_mesh_to_session.py --session-id <SESSION_ID> --user-mesh-npz user_mesh.npz --vertex-labels-npy vertex_labels.npy
```
