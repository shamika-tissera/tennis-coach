from pathlib import Path

WINDOW_BEFORE: float = 0.8
WINDOW_AFTER: float = 0.6
D_MAX_FORM: float = 1.0

_ROOT = Path(__file__).resolve().parents[1]


def _file_or_url(rel_path: str, url: str) -> str:
    """Use a local file if present; otherwise fall back to the given URL."""
    local_path = _ROOT / rel_path
    return str(local_path) if local_path.exists() else url


# Detector config/weights (RTMDet-tiny trained on COCO).
DET_CONFIG: str = _file_or_url(
    "configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
    "https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
)
DET_CHECKPOINT: str = _file_or_url(
    "checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
)

# Pose config/weights (COCO-WholeBody top-down R50).
POSE_CONFIG: str = _file_or_url(
    "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/"
    "td-hm_res50_8xb64-210e_coco-wholebody-256x192.py",
    "https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/"
    "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/"
    "td-hm_res50_8xb64-210e_coco-wholebody-256x192.py",
)
POSE_CHECKPOINT: str = _file_or_url(
    "checkpoints/res50_coco_wholebody_256x192-9e37ed88_20201004.pth",
    "https://download.openmmlab.com/mmpose/top_down/resnet/"
    "res50_coco_wholebody_256x192-9e37ed88_20201004.pth",
)

DET_SCORE_THR: float = 0.5
DEVICE: str = "cuda:0"
