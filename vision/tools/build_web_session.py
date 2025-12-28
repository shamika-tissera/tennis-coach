from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tennis_video.config as config
from tennis_video.web_session import build_web_session_payload, new_session_id, write_web_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a web-viewable session (user vs expert) and write it under data/web_sessions/."
    )
    parser.add_argument("--video", required=True, help="Path to user video.")
    parser.add_argument(
        "--t-impact",
        type=float,
        default=None,
        help="Impact time in seconds. If omitted, auto-estimate via wrist-speed peak.",
    )
    parser.add_argument(
        "--multi-shot",
        action="store_true",
        help="Detect multiple shots in a single clip (ignored if --t-impact is set).",
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=5,
        help="Maximum shots to extract when using --multi-shot.",
    )
    parser.add_argument(
        "--min-shot-separation-s",
        type=float,
        default=1.0,
        help="Minimum time separation between detected shots (seconds).",
    )
    parser.add_argument("--stroke-type", default=None, help="Stroke type label (optional).")
    parser.add_argument("--view", default=None, help="Optional view filter: topview or sideview.")
    parser.add_argument(
        "--expert-template",
        default=None,
        help="Optional explicit expert template .npz path; overrides template search.",
    )
    parser.add_argument(
        "--expert-root",
        default="expert_templates",
        help="Root directory containing expert templates.",
    )
    parser.add_argument(
        "--expert-manifest-csv",
        default="data/indoor_field_manifest_est.csv",
        help="CSV mapping expert stroke_id -> video_path,t_impact.",
    )
    parser.add_argument(
        "--sessions-dir",
        default="data/web_sessions",
        help="Output directory for sessions.",
    )
    parser.add_argument("--frames", type=int, default=120, help="Resampled playback frame count.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from tennis_video.pose_backend_mmpose import MMPoseBackend

    backend = MMPoseBackend(
        pose_config=config.POSE_CONFIG,
        pose_checkpoint=config.POSE_CHECKPOINT,
        det_config=config.DET_CONFIG,
        det_checkpoint=config.DET_CHECKPOINT,
        device=args.device,
        det_score_thr=args.det_score_thr,
    )

    if args.multi_shot and args.t_impact is None:
        from tennis_video.impact_detection import estimate_impact_times

        t_impacts = estimate_impact_times(
            args.video,
            backend,
            max_impacts=int(args.max_shots),
            min_separation_s=float(args.min_shot_separation_s),
        )
        base_id = new_session_id(args.video)
        sessions: list[str] = []
        for i, t_impact in enumerate(t_impacts, start=1):
            session_id = f"{base_id}_shot{i:02d}"
            payload = build_web_session_payload(
                pose_backend=backend,
                user_video_path=args.video,
                t_impact_user=float(t_impact),
                stroke_type=args.stroke_type,
                view=args.view,
                expert_template_path=args.expert_template,
                expert_root=Path(args.expert_root),
                expert_manifest_csv=Path(args.expert_manifest_csv),
                window_before=float(args.window_before),
                window_after=float(args.window_after),
                resample_len=int(args.frames),
            )
            payload["session_id"] = session_id
            payload["shot_index"] = i
            out = write_web_session(
                sessions_dir=Path(args.sessions_dir), session_id=session_id, payload=payload
            )
            sessions.append(session_id)
            print(f"Wrote session: {session_id}  t_impact={t_impact:.3f}s")
            print(out)
        print(f"Created {len(sessions)} sessions.")
        return

    session_id = new_session_id(args.video)
    payload = build_web_session_payload(
        pose_backend=backend,
        user_video_path=args.video,
        t_impact_user=args.t_impact,
        stroke_type=args.stroke_type,
        view=args.view,
        expert_template_path=args.expert_template,
        expert_root=Path(args.expert_root),
        expert_manifest_csv=Path(args.expert_manifest_csv),
        window_before=float(args.window_before),
        window_after=float(args.window_after),
        resample_len=int(args.frames),
    )
    payload["session_id"] = session_id
    out = write_web_session(
        sessions_dir=Path(args.sessions_dir), session_id=session_id, payload=payload
    )
    print(f"Wrote session: {session_id}")
    print(out)


if __name__ == "__main__":
    main()
