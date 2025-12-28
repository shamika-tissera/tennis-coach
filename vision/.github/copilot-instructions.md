# AI coding agent instructions for this repo

## Project overview
- This repo analyzes tennis strokes from video using OpenMMLab (MMDetection + MMPose).
- Core pipeline turns a raw video + approximate impact time into a pose-based time series (angles over time) and compares it to an expert template.
- The code is lightweight and script-oriented; keep new utilities similarly small and composable.

## Key modules and data flow
- `tennis_video/analyze_video.py`
  - Orchestrates the main analysis.
  - Uses `stroke_window.extract_video_window` to slice a small time window around a known impact time.
  - Uses `pose_backend_mmpose.MMPoseBackend` to turn video frames into pose angle vectors.
  - Produces a `PoseStroke` (see `stroke_types.PoseStroke`) and compares it to an expert stroke via `similarity_dtw.form_distance` and `form_scoring.form_similarity_score`.
- `tennis_video/config.py`
  - Central config for window size, maximum form distance, and OpenMMLab model configs/checkpoints and device.
  - When adding features, prefer threading new constants through this module rather than hard-coding paths or thresholds.
- `tennis_video/pose_backend_mmpose.py`
  - Wraps MMDetection + MMPose for single-person 2D pose.
  - Public surface:
    - `MMPoseBackend.keypoints_from_frame(frame_bgr) -> Optional[np.ndarray]` (K, 3 keypoints or None).
    - `MMPoseBackend.angle_vector_from_frame(frame_bgr) -> Optional[np.ndarray]`, which normalizes keypoints (`pose_processing`) and returns a 1D angle vector.
  - Any new backends should expose the same two methods so they can be dropped in to existing analysis code.
- `tennis_video/stroke_window.py`
  - `extract_video_window(video_path, t_impact, window_before, window_after)`
  - Relies on `video_io.iterate_frames` for `(index, t_frame, frame)` and filters frames in `[t_impact - window_before, t_impact + window_after]`.
  - Returns a list of `(tau, frame_bgr)` with `tau = t_frame - t_impact`.
- `tennis_video/stroke_types.py`
  - Defines `PoseStroke` dataclass with `stroke_id`, `stroke_type`, `tau` (1D array), `angles` (2D array [T, D]).
  - Provides `save_pose_stroke`/`load_pose_stroke` and an `ExpertLibrary` that reads all `*.npz` + sidecar JSON from a directory.
  - New stroke-like representations should mirror this simple dataclass + npz+JSON pattern.
- `tools/estimate_impact_times.py`
  - CLI utility to fill missing `t_impact` in a CSV manifest by detecting the frame of peak motion using optical flow + median filtering.
  - Inputs: CSV with `video_path,t_impact,stroke_type,stroke_id`. Outputs an updated CSV.
  - Use this to bootstrap `t_impact` values for new datasets before running pose analysis.

## Conventions & patterns
- **Types & style**
  - Python 3.10+ type hints are used (`from __future__ import annotations`). Maintain typing on new functions and keep return types precise (`Optional[...]`, `tuple[...]`, etc.).
  - Prefer small, pure helper functions (e.g., `median_filter`, `extract_video_window`) instead of embedding complex logic in CLIs.
- **Data formats**
  - Pose strokes: `.npz` files with arrays and a `.npz.json` sidecar storing `stroke_id` and `stroke_type`.
  - Manifests: CSV files with at least `video_path` and `t_impact` columns. Preserve additional columns when updating.
- **OpenMMLab integration**
  - Use `Config.fromfile` for pose configs and high-level `init_detector`/`init_model` helpers.
  - Detection results are filtered by a score threshold (`det_score_thr` from config). When changing thresholds, route them through `config.py`.

## Typical workflows
- **Estimate impact times for a dataset**
  - Use `tools/estimate_impact_times.py` to populate empty `t_impact` fields in a manifest CSV.
  - The script computes dense optical flow (Farnebäck), smooths it with a median filter (`--smooth-window`), and picks the largest peak within a configurable search region (`--search-start/--search-end`).
- **Analyze a user stroke vs expert template**
  - Build a `MMPoseBackend` using parameters from `tennis_video/config.py`.
  - Load an expert `PoseStroke` (e.g., from `expert_templates/*.npz`).
  - Call `analyze_video.analyze_stroke_against_expert(video_path, t_impact, stroke_type, expert_stroke, backend)` to get `(user_stroke, s_form, d_form)`.

## Guidance for AI changes
- When adding new analysis steps, prefer composing around `PoseStroke` (transform existing `tau`/`angles`) rather than inventing new core types.
- If you introduce new CLIs under `tools/`, follow the pattern in `estimate_impact_times.py`: small `parse_args`, pure helpers, and a `main()` guarded by `if __name__ == "__main__":`.
- Keep filesystem paths and magic numbers centralized in `tennis_video/config.py` or configuration arguments instead of inlined literals.
- Preserve compatibility with existing public functions (`build_user_pose_stroke`, `analyze_stroke_against_expert`, `MMPoseBackend` methods, `PoseStroke` and save/load helpers) unless explicitly refactoring call sites.
