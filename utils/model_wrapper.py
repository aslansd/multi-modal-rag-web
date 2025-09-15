import base64
import json
import requests
from typing import Iterator, List

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"

# ------------------------------------------------------------------
# 1. Streaming text-only via Ollama
# ------------------------------------------------------------------
def stream_ollama(prompt: str) -> Iterator[str]:
    """Text-only streaming with Ollama."""
    
    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": MODEL_NAME,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}]
        },
        stream=True,
    )
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8").removeprefix("data: ")
            try:
                content = json.loads(data)  # response is JSON per line
                yield content.get("message", {}).get("content", "")
            except Exception:
                continue

# ------------------------------------------------------------------
# 2. Streaming multimodal (text + images)
# ------------------------------------------------------------------
def stream_ollama_mm(text: str, images: List[str]) -> Iterator[str]:
    """Multimodal streaming with Ollama. Images will be base64-encoded."""
    
    image_payloads = []
    for path in images:
        with open(path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")
            image_payloads.append({"type": "image", "data": b64_img})

    messages = [{"role": "user", "content": [{"type": "text", "text": text}] + image_payloads}]

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True
        },
        stream=True,
    )

    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8").removeprefix("data: ")
            try:
                content = json.loads(data)
                yield content.get("message", {}).get("content", "")
            except Exception:
                continue