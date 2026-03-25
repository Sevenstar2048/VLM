from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import requests

from .rule_eval import VideoSafetyResult


class LLMSafetyEvaluator:
    """
    使用 OpenAI 兼容接口进行二次安全判别。
    需要环境变量:
    - LLM_API_KEY
    可选:
    - LLM_BASE_URL (默认 https://api.openai.com/v1)
    - LLM_MODEL (默认 gpt-4.1-mini)
    """

    def __init__(self, prompt_template_path: str):
        self.prompt_template_path = prompt_template_path
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "gpt-4.1-mini")

        template = Path(prompt_template_path).read_text(encoding="utf-8")
        self.template = template

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def _build_prompt(
        self,
        rule_result: VideoSafetyResult,
        keyframes: Dict[str, list[str]],
    ) -> str:
        payload = {
            "video_path": rule_result.video_path,
            "rule_result": asdict(rule_result),
            "keyframes": keyframes,
        }
        return self.template.replace("{{PAYLOAD_JSON}}", json.dumps(payload, ensure_ascii=False, indent=2))

    def evaluate(
        self,
        rule_result: VideoSafetyResult,
        keyframes: Dict[str, list[str]],
        timeout: int = 60,
    ) -> Optional[Dict[str, object]]:
        if not self.is_enabled():
            return None

        prompt = self._build_prompt(rule_result, keyframes)
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": "你是自动驾驶视频安全评测专家，只输出有效JSON。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # 兜底: 尝试提取第一个 JSON 对象
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(text[start : end + 1])
            else:
                raise

        return parsed
