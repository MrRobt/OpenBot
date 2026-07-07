"""Clients for Qwen, TTS, and Depth Anything 3 services."""

from __future__ import annotations

import base64
import logging
import os
import random
from typing import Any

import httpx
from schemas import Perception3D

logger = logging.getLogger(__name__)

MOCK_SERVICES = os.getenv("MOCK_SERVICES", "0") == "1"


class QwenClient:
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "Qwen2.5-Omni-3B-AWQ") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=60.0)
        self._mock = MOCK_SERVICES

    def decide(self, messages: list[dict[str, Any]]) -> str:
        if self._mock:
            # Simulate a decision based on the last user message text.
            text = ""
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text += item.get("text", "")

            hint = "CLEAR_AHEAD"
            if "BLOCKED_FRONT" in text:
                hint = "BLOCKED_FRONT"
            elif "FREE_LEFT" in text:
                hint = "FREE_LEFT"
            elif "FREE_RIGHT" in text:
                hint = "FREE_RIGHT"

            if "STOP" in text or "停车" in text:
                cmd = "STOP"
                reason = "mock: user asked to stop"
            elif hint == "BLOCKED_FRONT":
                cmd = "TURN_LEFT" if random.random() > 0.5 else "TURN_RIGHT"
                reason = "mock: obstacle ahead"
            elif "跟着" in text:
                cmd = "FOLLOW"
                reason = "mock: follow command"
            else:
                cmd = "FORWARD"
                reason = "mock: clear ahead"

            return (
                f'```json\n{{'
                f'"command": "{cmd}",'
                f'"param": 0.5,'
                f'"duration_ms": 800,'
                f'"tts_text": "收到，{cmd}",'
                f'"reason": "{reason}"'
                f'}}\n```'
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 256,
        }
        response = self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class TTSClient:
    def __init__(self, base_url: str = "http://localhost:8001") -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)
        self._mock = MOCK_SERVICES

    def synthesize(self, text: str, text_language: str = "zh") -> bytes:
        if self._mock:
            # Return a tiny valid WAV header (silent audio) as placeholder.
            return (
                b"RIFF\x26\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
                b"\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x02\x00\x00\x00\x00\x00"
            )

        payload = {"text": text, "text_language": text_language}
        response = self.client.post(f"{self.base_url}/tts", json=payload)
        response.raise_for_status()
        return response.content


class DA3Client:
    def __init__(self, base_url: str = "http://localhost:8002") -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def health(self) -> dict[str, Any]:
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def update_map(self, image_base64: str) -> dict[str, Any]:
        payload = {"image_base64": image_base64}
        response = self.client.post(f"{self.base_url}/map/update", json=payload)
        response.raise_for_status()
        return response.json()

    def navigate(self, target_angle_deg: float = 0.0) -> dict[str, Any]:
        payload = {"target_angle_deg": target_angle_deg}
        response = self.client.post(f"{self.base_url}/navigate", json=payload)
        response.raise_for_status()
        return response.json()

    def depth(self, image_base64: str) -> dict[str, Any]:
        payload = {"image_base64": image_base64}
        response = self.client.post(f"{self.base_url}/depth", json=payload)
        response.raise_for_status()
        return response.json()

    def to_perception_3d(self, nav_result: dict[str, Any], depth_result: dict[str, Any] | None = None) -> Perception3D:
        return Perception3D(
            depth_map_base64=depth_result.get("depth_map_base64") if depth_result else None,
            obstacles=nav_result.get("obstacles", []),
            navigation_hint=nav_result.get("navigation_hint", "CLEAR_AHEAD"),
            pose=nav_result.get("pose"),
        )
