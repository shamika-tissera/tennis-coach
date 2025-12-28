from __future__ import annotations
import sys
from typing import Optional, Tuple
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmengine import Config
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_model
from .pose_processing import compute_angle_vector, normalize_keypoints_2d


class MMPoseBackend:
    """Thin wrapper around MMDetection + MMPose for single-person pose."""

    def __init__(
        self,
        pose_config: str,
        pose_checkpoint: str,
        det_config: str,
        det_checkpoint: str,
        device: str = "cuda:0",
        det_score_thr: float = 0.5,
    ) -> None:
        self.device = device
        self.det_model = self._init_detector_with_fallback(det_config, det_checkpoint, device)
        self.pose_model = self._init_pose_with_fallback(pose_config, pose_checkpoint, self.device)
        self.det_score_thr = det_score_thr
        # Keep track of the person class index if available.
        self._person_label = self._infer_person_label()

    def _init_detector_with_fallback(self, config_path: str, checkpoint: str, device: str):
        """Initialize detector and fall back to CPU if CUDA is unavailable."""
        try:
            return init_detector(config_path, checkpoint, device=device)
        except Exception as exc:  # pragma: no cover - defensive
            if device.startswith("cuda"):
                print(
                    f"[MMPoseBackend] Detector init failed on {device} ({exc}); retrying on CPU.",
                    file=sys.stderr,
                )
                self.device = "cpu"
                return init_detector(config_path, checkpoint, device="cpu")
            raise

    def _init_pose_with_fallback(self, config_path: str, checkpoint: str, device: str):
        """Initialize pose model and fall back to CPU if CUDA is unavailable."""
        cfg = Config.fromfile(config_path)
        try:
            return init_pose_model(cfg, checkpoint, device=device)
        except Exception as exc:  # pragma: no cover - defensive
            if device.startswith("cuda"):
                print(
                    f"[MMPoseBackend] Pose init failed on {device} ({exc}); retrying on CPU.",
                    file=sys.stderr,
                )
                self.device = "cpu"
                return init_pose_model(cfg, checkpoint, device="cpu")
            raise

    def _infer_person_label(self) -> Optional[int]:
        """Return the dataset label index for the 'person' class if known."""
        classes = getattr(self.det_model, "dataset_meta", {}).get("classes")
        if classes:
            try:
                return list(classes).index("person")
            except ValueError:
                return None
        return None

    @staticmethod
    def _to_numpy(arr) -> Optional[np.ndarray]:
        """Convert torch/array-like to numpy without copying unnecessarily."""
        if arr is None:
            return None
        if hasattr(arr, "detach"):
            arr = arr.detach()
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        return np.asarray(arr)

    def _person_detections(
        self, det_results
    ) -> list[Tuple[float, float, float, float, float]]:
        """Convert detection output to a list of (x1, y1, x2, y2, score)."""
        boxes: list[Tuple[float, float, float, float, float]] = []

        # MMDetection 3.x returns a DetDataSample with pred_instances.
        instances = getattr(det_results, "pred_instances", None)
        if instances is not None:
            if len(instances) == 0:
                return boxes
            bboxes = self._to_numpy(getattr(instances, "bboxes", None))
            labels = self._to_numpy(getattr(instances, "labels", None))
            scores = self._to_numpy(getattr(instances, "scores", None))
            if bboxes is None:
                return boxes
            if scores is None:
                scores = np.ones((len(bboxes),), dtype=float)
            if labels is None:
                labels = np.zeros((len(bboxes),), dtype=int)
            target_label = self._person_label
            for bbox, label, score in zip(bboxes, labels, scores):
                if target_label is not None and int(label) != target_label:
                    continue
                if score < self.det_score_thr:
                    continue
                boxes.append(
                    (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                        float(score),
                    )
                )
            return boxes

        # mmdet 2.x style: list/tuple of numpy arrays per class.
        det_arrays = det_results[0] if isinstance(det_results, tuple) else det_results
        if isinstance(det_arrays, list) and det_arrays:
            class_idx = 0 if self._person_label is None else self._person_label
            if 0 <= class_idx < len(det_arrays):
                for bbox in det_arrays[class_idx]:
                    if len(bbox) < 5:
                        continue
                    score = float(bbox[4])
                    if score < self.det_score_thr:
                        continue
                    boxes.append(
                        (
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                            score,
                        )
                    )
        elif isinstance(det_arrays, np.ndarray):
            # Single-class detector returning an (N,5) array.
            for bbox in det_arrays:
                if len(bbox) < 5:
                    continue
                score = float(bbox[4])
                if score < self.det_score_thr:
                    continue
                boxes.append(
                    (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                        score,
                    )
                )

        return boxes

    def _instances_to_keypoints(
        self, instances, default_score: float
    ) -> Optional[np.ndarray]:
        """Extract (K,3) keypoints array from a PoseDataSample instances."""
        keypoints = self._to_numpy(getattr(instances, "keypoints", None))
        if keypoints is None or keypoints.size == 0:
            return None
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]

        scores = self._to_numpy(getattr(instances, "keypoint_scores", None))
        if scores is not None:
            if scores.ndim == 2:
                scores = scores[:, :, None]
            kp_scores = scores[0]
        else:
            kp_scores = np.full((keypoints.shape[1], 1), default_score, dtype=float)

        if kp_scores.shape[0] != keypoints.shape[1]:
            kp_scores = np.resize(kp_scores, (keypoints.shape[1], 1))

        return np.concatenate([keypoints[0], kp_scores], axis=-1)

    def keypoints_from_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Runs detector then pose estimation and returns keypoints (K,3) or None."""
        # Ensure the registries are scoped correctly for each library.
        init_default_scope("mmdet")
        det_results = inference_detector(self.det_model, frame_bgr)
        person_boxes = self._person_detections(det_results)
        if not person_boxes:
            return None

        # Sort detections by score so the first pose result corresponds to best box.
        person_boxes = sorted(person_boxes, key=lambda b: b[4], reverse=True)
        bboxes_xyxy = np.asarray([b[:4] for b in person_boxes], dtype=np.float32)

        init_default_scope(self.pose_model.cfg.get("default_scope", "mmpose"))
        pose_results = inference_topdown(
            self.pose_model, frame_bgr, bboxes_xyxy, bbox_format="xyxy"
        )
        if not pose_results:
            return None
        best_idx = 0
        best_sample = pose_results[best_idx]
        instances = getattr(best_sample, "pred_instances", None)
        if instances is None or len(instances) == 0:
            return None
        return self._instances_to_keypoints(instances, default_score=person_boxes[best_idx][4])

    def angle_vector_from_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns normalized angle vector for the best person in the frame."""
        keypoints = self.keypoints_from_frame(frame_bgr)
        if keypoints is None:
            return None
        coords_norm = normalize_keypoints_2d(keypoints)
        if coords_norm is None:
            return None
        return compute_angle_vector(coords_norm)
