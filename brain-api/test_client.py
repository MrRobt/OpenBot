"""Simple test client for Robot Brain API."""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image


def encode_image(path: str) -> str:
    img = Image.open(path).convert("RGB")
    # Resize to a reasonable input size.
    img = img.resize((224, 224))
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Test OpenBot Brain API")
    parser.add_argument("--server", default="http://localhost:8080", help="Brain API URL")
    parser.add_argument("--image", default="test.jpg", help="Path to test image")
    parser.add_argument("--command", default="前进", help="Voice/text command")
    parser.add_argument("--sonar", type=float, default=100.0, help="Sonar distance in cm")
    args = parser.parse_args()

    image_b64 = encode_image(args.image)
    payload = {
        "input": {
            "image_base64": image_b64,
            "sonar": args.sonar,
            "command_text": args.command,
        },
        "enable_3d": True,
    }

    start = time.time()
    response = httpx.post(f"{args.server}/decide", json=payload, timeout=60.0)
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    print(f"Latency: {elapsed * 1000:.1f}ms")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
