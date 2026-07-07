"""Robot Brain API: orchestrates Qwen, TTS, and Depth Anything 3."""

from __future__ import annotations

import base64
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from clients import DA3Client, QwenClient, TTSClient
from fastapi import FastAPI
from prompts import build_user_prompt, parse_decision
from pydantic import BaseModel, Field
from schemas import Perception3D, RobotDecision, RobotInput

import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brain_api")

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["qwen"] = QwenClient(
        base_url=os.getenv("QWEN_BASE_URL", "http://localhost:8000"),
        model=os.getenv("QWEN_MODEL", "Qwen2.5-Omni-3B-AWQ"),
    )
    _state["tts"] = TTSClient(base_url=os.getenv("TTS_BASE_URL", "http://localhost:8001"))
    _state["da3"] = DA3Client(base_url=os.getenv("DA3_BASE_URL", "http://localhost:8002"))
    yield
    _state.clear()


app = FastAPI(title="OpenBot Robot Brain API", lifespan=lifespan)


class DecisionRequest(BaseModel):
    input: RobotInput
    enable_3d: bool = Field(default=True)
    target_angle_deg: float = Field(default=0.0, ge=-180.0, le=180.0)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "qwen": _state.get("qwen") is not None,
        "tts": _state.get("tts") is not None,
        "da3": _state.get("da3") is not None,
    }


@app.post("/decide", response_model=RobotDecision)
def decide(req: DecisionRequest) -> RobotDecision:
    qwen_client = _state["qwen"]
    tts_client = _state["tts"]
    da3_client = _state["da3"]

    perception_3d: Perception3D | None = None
    nav_result: dict[str, Any] | None = None
    depth_result: dict[str, Any] | None = None

    if req.enable_3d:
        try:
            nav_result = da3_client.navigate(target_angle_deg=req.target_angle_deg)
            # Optionally also fetch depth map for debugging/visualization.
            depth_result = da3_client.depth(req.input.image_base64)
            perception_3d = da3_client.to_perception_3d(nav_result, depth_result)
        except Exception as exc:
            logger.warning("3D perception failed: %s", exc)
            perception_3d = Perception3D(navigation_hint="CLEAR_AHEAD")

    messages = build_user_prompt(req.input.model_dump(), perception_3d.model_dump() if perception_3d else None)
    raw_decision = qwen_client.decide(messages)
    decision = parse_decision(raw_decision)

    # Enforce safety override: sonar too close.
    if req.input.sonar >= 0 and req.input.sonar < 30:
        decision["command"] = "STOP"
        decision["param"] = 0.0
        decision["reason"] = f"safety override: sonar {req.input.sonar}cm"
        if not decision.get("tts_text"):
            decision["tts_text"] = "前方太近，停车"

    tts_audio_b64: str | None = None
    tts_text = decision.get("tts_text", "")
    if tts_text:
        try:
            audio_bytes = tts_client.synthesize(tts_text)
            tts_audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as exc:
            logger.warning("TTS failed: %s", exc)

    return RobotDecision(
        command=decision.get("command", "STOP"),
        param=float(decision.get("param", 0.0)),
        duration_ms=int(decision.get("duration_ms", 500)),
        tts_text=tts_text,
        tts_audio_base64=tts_audio_b64,
        reason=decision.get("reason", ""),
        perception_3d=perception_3d,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    args = parser.parse_args()

    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)
