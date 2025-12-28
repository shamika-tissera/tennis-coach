from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import tennis_video.config as config
from tennis_video.web_session import (
    build_web_session_payload,
    new_session_id,
    write_web_session,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sessions_dir() -> Path:
    root = _repo_root()
    env = os.getenv("TCVISION_WEB_SESSIONS_DIR")
    if env:
        return Path(env).expanduser()
    return root / "data" / "web_sessions"


def _static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


class CreateSessionRequest(BaseModel):
    video_path: str = Field(..., description="Path to user video.")
    t_impact: Optional[float] = Field(None, description="Impact time in seconds.")
    multi_shot: bool = Field(
        False, description="If true and t_impact is omitted, detect multiple shots in the clip."
    )
    max_shots: int = Field(5, ge=1, le=20, description="Maximum shots to extract when multi_shot.")
    min_shot_separation_s: float = Field(
        1.0, ge=0.0, le=10.0, description="Minimum time separation between detected shots."
    )
    stroke_type: Optional[str] = Field(None, description="Stroke type label, e.g., straight_shot.")
    view: Optional[str] = Field(None, description="Optional camera view filter: topview or sideview.")
    expert_template_path: Optional[str] = Field(
        None, description="Optional explicit expert template .npz path."
    )
    expert_root: str = Field("expert_templates", description="Directory containing expert templates.")
    expert_manifest_csv: str = Field(
        "data/indoor_field_manifest_est.csv",
        description="CSV mapping expert stroke_id -> video_path,t_impact.",
    )
    resample_len: int = Field(120, ge=10, le=600, description="Playback frame count.")
    window_before: float = Field(config.WINDOW_BEFORE, gt=0.0, le=5.0)
    window_after: float = Field(config.WINDOW_AFTER, gt=0.0, le=5.0)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: Optional[str] = None
    messages: Optional[list[ChatMessage]] = None


class GenerateMeshRequest(BaseModel):
    include_expert: bool = Field(True, description="If true, also generate the expert mesh when available.")
    clip_to_window: bool = Field(
        True,
        description="If true, clip to the stroke window (start/end inferred from tau around impact).",
    )


def _build_llm_messages(
    session: dict[str, Any], history: list[dict[str, str]]
) -> list[dict[str, str]]:
    metrics = session.get("metrics", {})
    feedback = session.get("feedback", [])
    expert = session.get("expert", {})
    user = session.get("user", {})

    context_lines = [
        "You are a tennis coach.",
        "Use the session context to answer the user's question about their stroke.",
        "Be concise, specific, and actionable.",
        "",
        f"Stroke type: {session.get('stroke_type')}",
        f"User video: {user.get('video_path')}",
        f"Expert template: {expert.get('template_path')}",
        f"Form similarity: {metrics.get('form_similarity')}",
        f"Form distance: {metrics.get('form_distance')}",
        "",
        "Auto-generated feedback:",
        *(f"- {line}" for line in feedback),
    ]
    system = "\n".join(context_lines)
    cleaned: list[dict[str, str]] = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        cleaned.append({"role": role, "content": content})

    return [{"role": "system", "content": system}, *cleaned]


def _call_openai_compatible_chat(messages: list[dict[str, str]]) -> str:
    """
    Call an OpenAI-compatible chat completions endpoint using env vars:
      - TCVISION_LLM_BASE_URL (default: https://integrate.api.nvidia.com/v1)
      - TCVISION_LLM_API_KEY (required)
      - TCVISION_LLM_MODEL   (default: openai/gpt-oss-20b)
    """
    import json

    import requests

    base_url = os.getenv("TCVISION_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    api_key = os.getenv("TCVISION_LLM_API_KEY")
    model = os.getenv("TCVISION_LLM_MODEL", "openai/gpt-oss-20b")
    if not api_key:
        raise RuntimeError("Missing TCVISION_LLM_API_KEY env var.")

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 1,
        "max_tokens": 800,
        "stream": False,
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unexpected LLM response shape: {data}") from exc


def _friendly_openmmlab_error(exc: BaseException) -> str:
    msg = str(exc)
    if "mmcv/_ext" in msg and "undefined symbol" in msg:
        try:
            import torch

            torch_ver = torch.__version__
        except Exception:
            torch_ver = "unknown"
        return (
            "OpenMMLab backend failed to load (MMCV C++ ops mismatch).\n"
            f"- Detected torch: {torch_ver}\n"
            "- Fix: reinstall MMCV built for your torch/CUDA build (or align torch to MMCV).\n"
            "  Example:\n"
            "    pip uninstall -y mmcv mmcv-full\n"
            "    pip install -U openmim\n"
            "    mim install mmcv\n"
            "  Then restart this server.\n"
            f"Raw error: {msg}"
        )
    return msg


def create_app() -> FastAPI:
    app = FastAPI(title="Tennis Coach Vision Web UI", version="0.1.0")

    sessions_dir = _sessions_dir()
    sessions_dir.mkdir(parents=True, exist_ok=True)

    static_dir = _static_dir()
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    app.mount("/sessions", StaticFiles(directory=str(sessions_dir)), name="sessions")

    backend_holder: dict[str, Any] = {}
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()

    def get_backend() -> Any:
        backend = backend_holder.get("backend")
        if backend is None:
            backend_name = (os.getenv("TCVISION_POSE_BACKEND", "mmpose") or "mmpose").strip().lower()
            if backend_name in {"mediapipe", "mp"}:
                from tennis_video.pose_backend_mediapipe import MediaPipeBackend

                backend = MediaPipeBackend()
            elif backend_name in {"mmpose"}:
                from tennis_video.pose_backend_mmpose import MMPoseBackend

                backend = MMPoseBackend(
                    pose_config=config.POSE_CONFIG,
                    pose_checkpoint=config.POSE_CHECKPOINT,
                    det_config=config.DET_CONFIG,
                    det_checkpoint=config.DET_CHECKPOINT,
                    device=os.getenv("TCVISION_DEVICE", config.DEVICE),
                    det_score_thr=float(os.getenv("TCVISION_DET_SCORE_THR", str(config.DET_SCORE_THR))),
                )
            else:  # auto
                mmpose_exc: Exception | None = None
                try:
                    from tennis_video.pose_backend_mmpose import MMPoseBackend

                    backend = MMPoseBackend(
                        pose_config=config.POSE_CONFIG,
                        pose_checkpoint=config.POSE_CHECKPOINT,
                        det_config=config.DET_CONFIG,
                        det_checkpoint=config.DET_CHECKPOINT,
                        device=os.getenv("TCVISION_DEVICE", config.DEVICE),
                        det_score_thr=float(
                            os.getenv("TCVISION_DET_SCORE_THR", str(config.DET_SCORE_THR))
                        ),
                    )
                except Exception as exc:
                    mmpose_exc = exc
                    try:
                        from tennis_video.pose_backend_mediapipe import MediaPipeBackend

                        backend = MediaPipeBackend()
                    except Exception:
                        raise mmpose_exc
            backend_holder["backend"] = backend
        return backend

    def _set_job(job_id: str, **fields: Any) -> None:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                return
            job.update(fields)
            job["updated_at"] = time.time()

    def _new_job(session_id: str) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        with jobs_lock:
            jobs[job_id] = {
                "job_id": job_id,
                "session_id": session_id,
                "status": "queued",
                "detail": "",
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
        return job_id

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        from tennis_video.mesh_generation import mesh_generator_config_from_env

        gen = mesh_generator_config_from_env()
        smpl_model = os.getenv("TCVISION_SMPL_MODEL")
        smpl_path = Path(smpl_model).expanduser() if smpl_model else None
        smpl_ok = bool(smpl_path and smpl_path.exists())
        available = bool(gen is not None or smpl_ok)
        hint = ""
        if not available:
            hint = (
                "Set TCVISION_SMPL_MODEL to your local SMPL model file (e.g. SMPL_NEUTRAL.pkl) "
                "to enable built-in one-click mesh generation from the session pose.\n"
                "Optional: set TCVISION_MESH_GENERATOR_CMD if you have an external exporter (e.g. MMHuman3D)."
            )
        return {
            "mesh_generation": {
                "available": available,
                "hint": hint,
                "clip_default": bool(int(os.getenv("TCVISION_MESH_CLIP_TO_WINDOW", "1"))),
            },
            "smpl_model": {
                "configured": bool(smpl_model and Path(smpl_model).expanduser().exists()),
                "path": smpl_model or "",
                "hint": (
                    ""
                    if smpl_model and Path(smpl_model).expanduser().exists()
                    else "Set TCVISION_SMPL_MODEL to enable body-part colors (labels) and face fallback."
                ),
            },
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (static_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/api/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for session_dir in sorted(sessions_dir.glob("*")):
            session_path = session_dir / "session.json"
            if not session_path.exists():
                continue
            try:
                session = json.loads(session_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append(
                {
                    "session_id": session_dir.name,
                    "created_at": session.get("created_at"),
                    "shot_index": session.get("shot_index"),
                    "stroke_type": session.get("stroke_type"),
                    "user_video_path": (session.get("user") or {}).get("video_path"),
                    "expert_template_path": (session.get("expert") or {}).get("template_path"),
                    "form_similarity": (session.get("metrics") or {}).get("form_similarity"),
                }
            )
        out.sort(key=lambda s: str(s.get("created_at") or ""), reverse=True)
        return out

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        session_path = sessions_dir / session_id / "session.json"
        if not session_path.exists():
            raise HTTPException(status_code=404, detail="Unknown session_id")
        try:
            return json.loads(session_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read session: {exc}") from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown job_id")
            return dict(job)

    @app.post("/api/sessions/{session_id}/mesh/generate")
    def generate_mesh(session_id: str, req: GenerateMeshRequest) -> dict[str, Any]:
        from tennis_video.mesh_generation import (
            MeshGenerationError,
            MeshGeneratorConfig,
            mesh_generator_config_from_env,
        )
        from tennis_video.mesh_pipeline import (
            generate_and_attach_smpl_mesh,
            generate_and_attach_smpl_mesh_from_session_joints,
        )

        session_path = sessions_dir / session_id / "session.json"
        if not session_path.exists():
            raise HTTPException(status_code=404, detail="Unknown session_id")

        smpl_model_env = os.getenv("TCVISION_SMPL_MODEL")
        smpl_model_path = Path(smpl_model_env).expanduser() if smpl_model_env else None
        if smpl_model_path is not None and not smpl_model_path.exists():
            smpl_model_path = None

        base_cfg = mesh_generator_config_from_env()
        cfg = (
            MeshGeneratorConfig(
                cmd_template=base_cfg.cmd_template,
                timeout_s=base_cfg.timeout_s,
                clip_to_window=bool(req.clip_to_window),
            )
            if base_cfg is not None
            else None
        )
        if smpl_model_path is None and cfg is None:
            raise HTTPException(
                status_code=400,
                detail="Mesh generation not configured. Set TCVISION_SMPL_MODEL (recommended) or TCVISION_MESH_GENERATOR_CMD.",
            )

        job_id = _new_job(session_id)

        def _runner() -> None:
            _set_job(job_id, status="running", detail="generating mesh…")
            try:
                if smpl_model_path is not None:
                    generate_and_attach_smpl_mesh_from_session_joints(
                        session_path=session_path,
                        smpl_model_path=smpl_model_path,
                        include_expert=bool(req.include_expert),
                    )
                elif cfg is not None:
                    generate_and_attach_smpl_mesh(
                        repo_root=_repo_root(),
                        session_path=session_path,
                        include_expert=bool(req.include_expert),
                        generator_cfg=cfg,
                        smpl_model_path=smpl_model_path,
                    )
                else:
                    raise MeshGenerationError(
                        "No mesh generation method available. Set TCVISION_SMPL_MODEL or TCVISION_MESH_GENERATOR_CMD."
                    )
            except MeshGenerationError as exc:
                _set_job(job_id, status="error", error=str(exc), detail="failed")
                return
            except Exception as exc:
                _set_job(job_id, status="error", error=str(exc), detail="failed")
                return
            _set_job(job_id, status="done", detail="attached")

        threading.Thread(target=_runner, daemon=True).start()
        return {"job_id": job_id}

    @app.post("/api/sessions")
    def create_session(req: CreateSessionRequest) -> dict[str, Any]:
        expert_root = (_repo_root() / req.expert_root).resolve()
        manifest_csv = (_repo_root() / req.expert_manifest_csv).resolve()
        if not Path(req.video_path).exists():
            raise HTTPException(status_code=400, detail=f"Video not found: {req.video_path}")
        if req.expert_template_path and not Path(req.expert_template_path).exists():
            raise HTTPException(
                status_code=400, detail=f"Expert template not found: {req.expert_template_path}"
            )

        try:
            backend = get_backend()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=_friendly_openmmlab_error(exc)) from exc
        if req.multi_shot and req.t_impact is None:
            from tennis_video.impact_detection import estimate_impact_times

            try:
                t_impacts = estimate_impact_times(
                    req.video_path,
                    backend,
                    max_impacts=req.max_shots,
                    min_separation_s=req.min_shot_separation_s,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=_friendly_openmmlab_error(exc)) from exc

            base_id = new_session_id(req.video_path)
            created: list[dict[str, Any]] = []
            for i, t_impact in enumerate(t_impacts, start=1):
                session_id = f"{base_id}_shot{i:02d}"
                try:
                    payload = build_web_session_payload(
                        pose_backend=backend,
                        user_video_path=req.video_path,
                        t_impact_user=float(t_impact),
                        stroke_type=req.stroke_type,
                        view=req.view,
                        expert_template_path=req.expert_template_path,
                        expert_root=expert_root,
                        expert_manifest_csv=manifest_csv,
                        window_before=req.window_before,
                        window_after=req.window_after,
                        resample_len=req.resample_len,
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=500, detail=_friendly_openmmlab_error(exc)
                    ) from exc
                payload["session_id"] = session_id
                payload["shot_index"] = i
                try:
                    write_web_session(
                        sessions_dir=sessions_dir, session_id=session_id, payload=payload
                    )
                except FileExistsError:
                    raise HTTPException(status_code=409, detail="Session already exists.")
                except Exception as exc:
                    raise HTTPException(
                        status_code=500, detail=f"Failed to write session: {exc}"
                    ) from exc
                created.append({"session_id": session_id, "t_impact": float(t_impact)})

            return {"sessions": created}

        session_id = new_session_id(req.video_path)
        try:
            payload = build_web_session_payload(
                pose_backend=backend,
                user_video_path=req.video_path,
                t_impact_user=req.t_impact,
                stroke_type=req.stroke_type,
                view=req.view,
                expert_template_path=req.expert_template_path,
                expert_root=expert_root,
                expert_manifest_csv=manifest_csv,
                window_before=req.window_before,
                window_after=req.window_after,
                resample_len=req.resample_len,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=_friendly_openmmlab_error(exc)) from exc

        payload["session_id"] = session_id
        try:
            write_web_session(sessions_dir=sessions_dir, session_id=session_id, payload=payload)
        except FileExistsError:
            raise HTTPException(status_code=409, detail="Session already exists.")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to write session: {exc}") from exc

        return {"session_id": session_id}

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        session_path = sessions_dir / req.session_id / "session.json"
        if not session_path.exists():
            raise HTTPException(status_code=404, detail="Unknown session_id")
        session = json.loads(session_path.read_text(encoding="utf-8"))

        if req.messages is not None and len(req.messages) > 0:
            history = [{"role": m.role, "content": m.content} for m in req.messages][-16:]
        else:
            if not req.message:
                raise HTTPException(status_code=400, detail="Provide message or messages.")
            history = [{"role": "user", "content": req.message}]

        messages = _build_llm_messages(session, history)
        try:
            answer = _call_openai_compatible_chat(messages)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"answer": answer}

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "webapp.server:app",
        host=os.getenv("TCVISION_HOST", "127.0.0.1"),
        port=int(os.getenv("TCVISION_PORT", "8000")),
        reload=bool(int(os.getenv("TCVISION_RELOAD", "1"))),
    )


if __name__ == "__main__":
    main()
