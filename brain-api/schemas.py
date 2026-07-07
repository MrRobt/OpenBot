"""Pydantic request/response schemas for Robot Brain API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RobotInput(BaseModel):
    image_base64: str
    sonar: float = Field(default=-1.0, ge=-1.0, le=400.0)
    left_rpm: float = 0.0
    right_rpm: float = 0.0
    battery: float = Field(default=12.0, ge=0.0, le=30.0)
    command_text: str = ""


class Perception3D(BaseModel):
    depth_map_base64: str | None = None
    obstacles: list[dict[str, Any]] = Field(default_factory=list)
    navigation_hint: str = "CLEAR_AHEAD"
    pose: dict[str, float] | None = None


class RobotDecision(BaseModel):
    command: str
    param: float = Field(default=0.0, ge=0.0, le=1.0)
    duration_ms: int = Field(default=500, ge=200, le=2000)
    tts_text: str = ""
    tts_audio_base64: str | None = None
    reason: str = ""
    perception_3d: Perception3D | None = None
