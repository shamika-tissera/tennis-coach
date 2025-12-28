from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class MeshGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MeshGeneratorConfig:
    cmd_template: str
    timeout_s: int = 60 * 60
    clip_to_window: bool = True


def _resolve_video_path(video_path: str | Path, *, repo_root: Path) -> Path:
    p = Path(video_path).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _format_cmd(cmd: list[str], *, video: Path, out_npz: Path, start_s: float, end_s: float) -> list[str]:
    mapping = {
        "video": str(video),
        "out_npz": str(out_npz),
        "start": f"{float(start_s):.6f}",
        "end": f"{float(end_s):.6f}",
    }
    try:
        return [part.format(**mapping) for part in cmd]
    except KeyError as exc:
        raise MeshGenerationError(
            f"Mesh generator cmd uses an unknown placeholder: {exc}. "
            "Supported: {video} {out_npz} {start} {end}"
        ) from exc


def _clip_video_window_opencv(
    *,
    src_video: Path,
    dst_video: Path,
    start_s: float,
    end_s: float,
) -> Path:
    import cv2

    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        cap.release()
        raise MeshGenerationError(f"Could not open video for clipping: {src_video}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else 30.0

        ok = cap.set(cv2.CAP_PROP_POS_MSEC, float(start_s) * 1000.0)
        if not ok:
            cap.release()
            raise MeshGenerationError("OpenCV could not seek to start time for clipping.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        if width <= 0 or height <= 0:
            ret, frame = cap.read()
            if not ret:
                raise MeshGenerationError("Failed to read first frame while clipping.")
            height, width = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_MSEC, float(start_s) * 1000.0)

        dst_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") if dst_video.suffix.lower() == ".mp4" else cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(dst_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise MeshGenerationError(f"Failed to open VideoWriter: {dst_video}")

        try:
            while True:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms and pos_ms / 1000.0 > float(end_s) + 1e-3:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
        finally:
            writer.release()
    finally:
        cap.release()

    if not dst_video.exists() or dst_video.stat().st_size <= 0:
        raise MeshGenerationError("Clipped video was not written.")
    return dst_video


def run_external_mesh_generator(
    *,
    video_path: Path,
    out_npz: Path,
    start_s: float,
    end_s: float,
    cfg: MeshGeneratorConfig,
) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    cmd_parts = shlex.split(cfg.cmd_template)
    if not cmd_parts:
        raise MeshGenerationError("TCVISION_MESH_GENERATOR_CMD is empty.")

    input_video = video_path
    tmp_dir_ctx = None
    if cfg.clip_to_window:
        try:
            tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="tcvision_mesh_clip_")
            tmp_dir = Path(tmp_dir_ctx.name)
            clipped = tmp_dir / "clip.mp4"
            input_video = _clip_video_window_opencv(
                src_video=video_path, dst_video=clipped, start_s=start_s, end_s=end_s
            )
        except Exception:
            if tmp_dir_ctx is not None:
                tmp_dir_ctx.cleanup()
            input_video = video_path
            tmp_dir_ctx = None

    try:
        cmd = _format_cmd(cmd_parts, video=input_video, out_npz=out_npz, start_s=start_s, end_s=end_s)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s,
            check=False,
        )
    finally:
        if tmp_dir_ctx is not None:
            tmp_dir_ctx.cleanup()

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"exit={proc.returncode}"
        raise MeshGenerationError(f"Mesh generator failed: {detail}")

    if not out_npz.exists() or out_npz.stat().st_size <= 0:
        raise MeshGenerationError("Mesh generator did not produce output NPZ.")


def mesh_generator_config_from_env() -> Optional[MeshGeneratorConfig]:
    cmd = os.getenv("TCVISION_MESH_GENERATOR_CMD")
    if not cmd:
        return None
    timeout_s = int(os.getenv("TCVISION_MESH_GENERATOR_TIMEOUT_S", str(60 * 60)))
    clip = bool(int(os.getenv("TCVISION_MESH_CLIP_TO_WINDOW", "1")))
    return MeshGeneratorConfig(cmd_template=cmd, timeout_s=timeout_s, clip_to_window=clip)

