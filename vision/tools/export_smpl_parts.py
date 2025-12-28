from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_video.smpl_parts import write_vertex_parts_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-vertex SMPL body-part labels from an SMPL model file by using blend weights."
        )
    )
    parser.add_argument(
        "--smpl-model",
        required=True,
        help="Path to SMPL model (.pkl or .npz) containing 'weights' (V,24).",
    )
    parser.add_argument(
        "--out-labels",
        default="smpl_vertex_labels.npy",
        help="Output .npy file containing labels (V,) uint16.",
    )
    parser.add_argument(
        "--out-meta",
        default="smpl_parts_meta.json",
        help="Output JSON containing {'parts': [...], 'palette': [[r,g,b],...]}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_labels, out_meta = write_vertex_parts_assets(
        Path(args.smpl_model),
        out_labels=Path(args.out_labels),
        out_meta=Path(args.out_meta),
    )
    print(f"Wrote labels: {out_labels}")
    print(f"Wrote meta: {out_meta}")


if __name__ == "__main__":
    main()
