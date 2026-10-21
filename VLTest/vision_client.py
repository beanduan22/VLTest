from __future__ import annotations

import base64
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def image_data(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return mime, data


class VisionClient:
    def __init__(self, model, base_url, api_key="", *, provider="openai", log=None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider = provider
        self.log = log
        self.role = "discovery"
        self.calls = 0

    def request(self, prompt, images, *, max_tokens=128):
        if self.provider == "gemini":
            parts = [{"text": prompt}]
            for path in images:
                mime, data = image_data(path)
                parts.append({"inline_data": {"mime_type": mime, "data": data}})
            return {"contents": [{"role": "user", "parts": parts}],
                    "generationConfig": {"temperature": 0,
                                         "maxOutputTokens": max_tokens,
                                         "thinkingConfig": {"thinkingBudget": 0}}}
        content = [{"type": "text", "text": prompt}]
        for path in images:
            mime, data = image_data(path)
            content.append({"type": "image_url", "image_url": {
                "url": f"data:{mime};base64,{data}", "detail": "high"}})
        return {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}]}

    def complete(self, prompt, images, *, max_tokens=128):
        payload = self.request(prompt, images, max_tokens=max_tokens)
        headers = {"Content-Type": "application/json"}
        if self.provider == "gemini":
            url = f"{self.base_url}/models/{self.model}:generateContent"
            headers["x-goog-api-key"] = self.api_key
        else:
            url = f"{self.base_url}/chat/completions"
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(url, data=json.dumps(payload).encode(), headers=headers)
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
        self.calls += 1
        if self.provider == "gemini":
            candidates = result.get("candidates", [])
            output = "".join(part.get("text", "")
                             for part in candidates[0].get("content", {}).get("parts", [])
                             if not part.get("thought")) if candidates else ""
        else:
            output = result["choices"][0]["message"].get("content") or ""
        if self.log is not None:
            self.log.write(json.dumps({
                "date": datetime.now(timezone.utc).isoformat(), "role": self.role,
                "model": self.model, "request": payload, "response": result,
            }, ensure_ascii=False) + "\n")
            self.log.flush()
        return output
