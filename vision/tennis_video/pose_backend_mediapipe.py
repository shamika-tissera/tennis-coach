from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class _MediaPipeCfg:
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class MediaPipeBackend:
    """Single-person pose backend using MediaPipe Pose (COCO-17 output)."""

    def __init__(self, *, cfg: Optional[_MediaPipeCfg] = None) -> None:
        self.cfg = cfg or _MediaPipeCfg()
        pose_mod = _import_mediapipe_pose_module()
        self._pose_mod = pose_mod
        self._pose = pose_mod.Pose(
            static_image_mode=False,
            model_complexity=int(self.cfg.model_complexity),
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=float(self.cfg.min_detection_confidence),
            min_tracking_confidence=float(self.cfg.min_tracking_confidence),
        )

    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass

    @staticmethod
    def _as_score(landmark) -> float:
        v = getattr(landmark, "visibility", None)
        if v is None:
            v = getattr(landmark, "presence", 0.0)
        try:
            return float(v)
        except Exception:
            return 0.0

    def keypoints_from_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Return COCO-17 keypoints as (17,3) in pixel coordinates: (x,y,score).
        """
        import cv2

        if frame_bgr is None:
            return None
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if results is None or results.pose_landmarks is None:
            return None

        lm = results.pose_landmarks.landmark
        idx = self._pose_mod.PoseLandmark

        mapping = [
            idx.NOSE,
            idx.LEFT_EYE,
            idx.RIGHT_EYE,
            idx.LEFT_EAR,
            idx.RIGHT_EAR,
            idx.LEFT_SHOULDER,
            idx.RIGHT_SHOULDER,
            idx.LEFT_ELBOW,
            idx.RIGHT_ELBOW,
            idx.LEFT_WRIST,
            idx.RIGHT_WRIST,
            idx.LEFT_HIP,
            idx.RIGHT_HIP,
            idx.LEFT_KNEE,
            idx.RIGHT_KNEE,
            idx.LEFT_ANKLE,
            idx.RIGHT_ANKLE,
        ]

        out = np.zeros((17, 3), dtype=np.float32)
        for i, m in enumerate(mapping):
            p = lm[int(m)]
            out[i, 0] = float(p.x) * float(w)
            out[i, 1] = float(p.y) * float(h)
            out[i, 2] = float(self._as_score(p))
        return out


def _import_mediapipe_pose_module() -> Any:
    """
    Import MediaPipe Pose without importing mediapipe.tasks (tensorflow).

    Newer `mediapipe` wheels import `mediapipe.tasks` at package import time,
    which can fail if TensorFlow/protobuf pins are incompatible. This helper
    falls back to a "lite" import path that bypasses `mediapipe/__init__.py` and
    imports `mediapipe.python.solutions.pose` directly.
    """
    try:
        import mediapipe as mp  # type: ignore

        pose_mod = getattr(getattr(mp, "solutions"), "pose", None)
        if pose_mod is None:
            raise AttributeError("mediapipe.solutions.pose is missing")
        return pose_mod
    except Exception as exc:
        try:
            return _import_mediapipe_pose_lite()
        except Exception as exc2:
            raise RuntimeError(
                "Failed to import MediaPipe Pose. "
                "If you're seeing TensorFlow/protobuf errors, try upgrading protobuf "
                "(e.g. `pip install -U protobuf`) or installing a compatible mediapipe version.\n"
                f"Original error: {exc}\n"
                f"Lite import error: {exc2}"
            ) from exc


def _import_mediapipe_pose_lite() -> Any:
    import importlib
    import importlib.machinery
    import importlib.util
    import re
    import sys
    import types
    from pathlib import Path

    spec = importlib.util.find_spec("mediapipe")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("mediapipe is not installed.")

    pkg_paths = [str(p) for p in spec.submodule_search_locations]

    # If a previous import failed mid-way, ensure we replace the broken entry.
    sys.modules.pop("mediapipe", None)

    pkg = types.ModuleType("mediapipe")
    pkg.__path__ = pkg_paths  # type: ignore[attr-defined]
    pkg.__package__ = "mediapipe"
    pkg.__spec__ = importlib.machinery.ModuleSpec("mediapipe", loader=None, is_package=True)
    sys.modules["mediapipe"] = pkg

    # Best-effort: provide the commonly used `mediapipe.solutions` and
    # `mediapipe.__version__` attributes without importing `mediapipe.tasks`.
    try:
        init_py = Path(pkg_paths[0]) / "__init__.py"
        if init_py.exists():
            txt = init_py.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"__version__\s*=\s*['\\\"]([^'\\\"]+)['\\\"]", txt)
            if m:
                setattr(pkg, "__version__", m.group(1))
    except Exception:
        pass

    try:
        solutions_pkg = importlib.import_module("mediapipe.python.solutions")
        setattr(pkg, "solutions", solutions_pkg)
    except Exception:
        pass

    pose_mod = importlib.import_module("mediapipe.python.solutions.pose")
    return pose_mod
