from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class PoseStroke:
    """Pose-based representation of a stroke."""
    stroke_id: str
    stroke_type: str
    tau: np.ndarray
    angles: np.ndarray


def save_pose_stroke(stroke: PoseStroke, path: str) -> None:
    """Saves PoseStroke to npz with metadata json sidecar."""
    np.savez_compressed(path, tau=stroke.tau, angles=stroke.angles)
    meta = {"stroke_id": stroke.stroke_id, "stroke_type": stroke.stroke_type}
    with open(f"{path}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_pose_stroke(path: str) -> PoseStroke:
    """Loads PoseStroke from npz and json metadata."""
    data = np.load(path)
    with open(f"{path}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    tau = np.asarray(data["tau"], dtype=float)
    angles = np.asarray(data["angles"], dtype=float)
    return PoseStroke(
        stroke_id=str(meta.get("stroke_id", os.path.basename(path))),
        stroke_type=str(meta.get("stroke_type", "unknown")),
        tau=tau,
        angles=angles,
    )


@dataclass
class ExpertLibrary:
    """Collection of expert template strokes."""
    strokes: Dict[str, PoseStroke]


def load_expert_library_from_dir(dir_path: str) -> ExpertLibrary:
    """Loads all PoseStroke templates from a directory."""
    strokes: Dict[str, PoseStroke] = {}
    for name in os.listdir(dir_path):
        if not name.endswith(".npz"):
            continue
        npz_path = os.path.join(dir_path, name)
        json_path = f"{npz_path}.json"
        if not os.path.isfile(json_path):
            continue
        stroke = load_pose_stroke(npz_path)
        key = stroke.stroke_type or os.path.splitext(name)[0]
        strokes[key] = stroke
    return ExpertLibrary(strokes=strokes)
