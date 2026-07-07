"""System prompt and response parsing for the robot brain."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the high-level brain of a small robot car.
You receive a camera image, ultrasonic distance, wheel speeds, battery voltage,
and optional 3D perception information (obstacles, navigation hint).

Your job is to output ONE high-level driving command as valid JSON.

Allowed commands:
STOP, FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, AVOID_LEFT, AVOID_RIGHT,
FOLLOW, ROTATE_LEFT, ROTATE_RIGHT, APPROACH, EXPLORE.

Output format:
{
  "command": "TURN_LEFT",
  "param": 0.5,
  "duration_ms": 800,
  "tts_text": "左转绕行",
  "reason": "obstacle on the right"
}

Rules:
- param must be between 0.0 and 1.0.
- duration_ms should be between 200 and 2000.
- tts_text should be a short Chinese phrase.
- If sonar distance is very small (< 30cm), prefer STOP or AVOID.
- Use 3D perception hint when available: FREE_LEFT means the left side is clear,
  FREE_RIGHT means the right side is clear, BLOCKED_FRONT means cannot go straight.
"""


def build_user_prompt(input_data: dict[str, Any], perception_3d: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Build the multimodal user prompt for Qwen."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_data['image_base64']}"}},
    ]

    text = (
        f"超声波距离: {input_data.get('sonar', -1)} cm\n"
        f"左轮转速: {input_data.get('left_rpm', 0)}\n"
        f"右轮转速: {input_data.get('right_rpm', 0)}\n"
        f"电池电压: {input_data.get('battery', 12)} V\n"
        f"用户指令: {input_data.get('command_text', '')}\n"
    )

    if perception_3d:
        hint = perception_3d.get("navigation_hint", "CLEAR_AHEAD")
        obstacles = perception_3d.get("obstacles", [])
        text += f"\n3D 导航提示: {hint}\n"
        if obstacles:
            text += "附近障碍物:\n"
            for obs in obstacles[:5]:
                text += f"- {obs.get('distance_m', 0):.2f}m, 方位 {obs.get('angle_deg', 0):.0f}°, 宽度 {obs.get('width_m', 0):.2f}m\n"

    content.append({"type": "text", "text": text})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def parse_decision(raw: str) -> dict[str, Any]:
    """Extract JSON decision from Qwen text response."""
    raw = raw.strip()

    # Try to find JSON block.
    if "```json" in raw:
        match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1)
    elif "```" in raw:
        match = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract the first JSON object.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON: %s", exc)

    logger.warning("Could not parse decision from: %s", raw[:200])
    return {
        "command": "STOP",
        "param": 0.0,
        "duration_ms": 500,
        "tts_text": "我没听懂，先停下",
        "reason": "parse error",
    }
