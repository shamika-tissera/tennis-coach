from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_video import (
    MMPoseBackend,
    analyze_video,
    config,
    estimate_impact_time,
    estimate_impact_time_tracknet_from_detections,
    load_tracknet_csv,
    generate_feedback,
)
from tennis_video.stroke_types import load_pose_stroke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a user stroke to an expert template and report form similarity."
    )
    parser.add_argument("--video-path", required=True, help="Path to user video.")
    parser.add_argument(
        "--t-impact",
        type=float,
        help="Impact time in seconds. If omitted, auto-estimate via wrist-speed peak.",
    )
    parser.add_argument(
        "--tracknet-csv",
        help="Path to TrackNet detections CSV (frame,x,y,score) for impact estimation.",
    )
    parser.add_argument(
        "--stroke-type",
        required=False,
        help="Stroke type label (e.g., straight_shot). Required if --expert is not set.",
    )
    parser.add_argument(
        "--expert",
        help="Path to expert template .npz; overrides automatic selection by stroke type.",
    )
    parser.add_argument(
        "--expert-root",
        default="expert_templates",
        help="Directory containing expert templates (used when --expert is not provided).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Report top-k matches when scanning all templates of a stroke type.",
    )
    parser.add_argument(
        "--device",
        default=config.DEVICE,
        help=f"Device for models (default: {config.DEVICE}). Use 'cpu' if no GPU.",
    )
    parser.add_argument(
        "--det-score-thr",
        type=float,
        default=config.DET_SCORE_THR,
        help="Detection score threshold for person boxes.",
    )
    parser.add_argument(
        "--window-before",
        type=float,
        default=config.WINDOW_BEFORE,
        help="Seconds before impact to include.",
    )
    parser.add_argument(
        "--window-after",
        type=float,
        default=config.WINDOW_AFTER,
        help="Seconds after impact to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expert is None and not args.stroke_type:
        raise SystemExit("Provide --stroke-type when --expert is not set.")

    backend = MMPoseBackend(
        pose_config=config.POSE_CONFIG,
        pose_checkpoint=config.POSE_CHECKPOINT,
        det_config=config.DET_CONFIG,
        det_checkpoint=config.DET_CHECKPOINT,
        device=args.device,
        det_score_thr=args.det_score_thr,
    )
    t_impact = args.t_impact
    if t_impact is None and args.tracknet_csv:
        detections = load_tracknet_csv(args.tracknet_csv)
        t_impact = estimate_impact_time_tracknet_from_detections(
            args.video_path, detections
        )
        print(f"TrackNet-estimated impact time: {t_impact:.3f}s")
    elif t_impact is None:
        t_impact = estimate_impact_time(args.video_path, backend)
        print(f"Auto-estimated impact time: {t_impact:.3f}s")

    # Build user stroke once.
    user_stroke = analyze_video.build_user_pose_stroke(
        video_path=args.video_path,
        t_impact=t_impact,
        stroke_id="user_stroke",
        stroke_type=args.stroke_type or "unknown",
        pose_backend=backend,
        window_before=config.WINDOW_BEFORE,
        window_after=config.WINDOW_AFTER,
    )

    # Compare to either a specific expert or all matching templates.
    matches = []
    if args.expert:
        expert_stroke = load_pose_stroke(args.expert)
        d_form = analyze_video.form_distance(user_stroke, expert_stroke)
        s_form = analyze_video.form_similarity_score(d_form, config.D_MAX_FORM)
        matches.append((args.expert, expert_stroke.stroke_type, s_form, d_form, expert_stroke))
    else:
        candidates = sorted(Path(args.expert_root).glob("**/*.npz"))
        for c in candidates:
            try:
                stroke = load_pose_stroke(str(c))
            except Exception:
                continue
            if stroke.stroke_type != args.stroke_type:
                continue
            d_form = analyze_video.form_distance(user_stroke, stroke)
            s_form = analyze_video.form_similarity_score(d_form, config.D_MAX_FORM)
            matches.append((str(c), stroke.stroke_type, s_form, d_form, stroke))

    if not matches:
        raise SystemExit("No expert templates matched the criteria.")

    matches.sort(key=lambda x: x[3])  # sort by distance ascending
    top_k = matches[: max(1, args.top_k)]

    print(f"User stroke frames: {len(user_stroke.tau)}, angles shape: {user_stroke.angles.shape}")
    print("Top matches:")
    for path, stype, s_form, d_form, _ in top_k:
        print(f"- {Path(path).name} [{stype}]  sim={s_form:.3f}  dist={d_form:.3f}")

    best_path, best_type, best_sim, best_dist, best_expert = top_k[0]
    print("\nBest match feedback:")
    for line in generate_feedback(user_stroke, best_expert):
        print(f"- {line}")


if __name__ == "__main__":
    main()
