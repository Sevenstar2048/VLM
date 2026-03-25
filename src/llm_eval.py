from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict, Optional

import requests


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
        self.backend = os.getenv("LLM_BACKEND", "api").strip().lower()
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
        self.max_images = int(os.getenv("LLM_MAX_IMAGES", "24"))
        self.image_detail = os.getenv("LLM_IMAGE_DETAIL", "low")

        template = Path(prompt_template_path).read_text(encoding="utf-8")
        self.template = template

    def is_enabled(self) -> bool:
        if self.backend in {"api", "openai_compatible"}:
            return bool(self.api_key)
        if self.backend == "ollama":
            return True
        return False

    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    @staticmethod
    def _guess_mime(image_path: str) -> str:
        suffix = Path(image_path).suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        if suffix == ".webp":
            return "image/webp"
        return "image/jpeg"

    def _ordered_image_paths(self, keyframes: Dict[str, list[str]]) -> list[str]:
        raw = keyframes.get("raw", [])
        det = keyframes.get("det", [])
        gen = keyframes.get("gen", [])
        count = min(len(raw), len(det), len(gen))

        ordered: list[str] = []
        for i in range(count):
            # 按时间顺序组织: raw(i) -> det(i) -> gen(i)
            ordered.extend([raw[i], det[i], gen[i]])

        return ordered[: self.max_images]

    def _build_prompt(
        self,
        video_path: str,
        keyframes: Dict[str, list[str]],
        rule_context: Optional[Dict[str, object]] = None,
    ) -> str:
        payload = {
            "video_path": video_path,
            "multi_view_layout": {
                "columns": 6,
                "rows": 3,
                "row_meaning": {
                    "raw": "原始视频行",
                    "det": "检测结果行",
                    "gen": "生成视频行",
                },
                "camera_order": "每张图从左到右为 cam0..cam5，方位顺序在同一数据集内固定",
            },
            "frame_groups": {
                "raw": len(keyframes.get("raw", [])),
                "det": len(keyframes.get("det", [])),
                "gen": len(keyframes.get("gen", [])),
            },
            "image_order": "按时间顺序重复 [raw_t, det_t, gen_t]",
            "rule_context": rule_context or {},
        }
        return self.template.replace("{{PAYLOAD_JSON}}", json.dumps(payload, ensure_ascii=False, indent=2))

    def evaluate(
        self,
        video_path: str,
        keyframes: Dict[str, list[str]],
        rule_context: Optional[Dict[str, object]] = None,
        timeout: int = 60,
    ) -> Optional[Dict[str, object]]:
        if not self.is_enabled():
            return None

        ordered_images = self._ordered_image_paths(keyframes)
        if not ordered_images:
            return {
                "semantic_unsafe": 0,
                "logical_unsafe": 0,
                "decision_unsafe": 0,
                "unsafe": 0,
                "confidence": 0.0,
                "reason": "未找到可用关键帧",
            }

        prompt = self._build_prompt(video_path, keyframes, rule_context=rule_context)
        if self.backend in {"api", "openai_compatible"}:
            text = self._evaluate_openai_compatible(prompt, ordered_images, timeout)
        elif self.backend == "ollama":
            text = self._evaluate_ollama(prompt, ordered_images, timeout)
        else:
            raise RuntimeError(f"不支持的 LLM_BACKEND: {self.backend}")

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

    def _evaluate_openai_compatible(self, prompt: str, ordered_images: list[str], timeout: int) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        user_content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for image_path in ordered_images:
            encoded = self._encode_image_base64(image_path)
            mime = self._guess_mime(image_path)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{encoded}",
                        "detail": self.image_detail,
                    },
                }
            )

        body = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": "你是自动驾驶视频安全评测专家。你将收到按时间排序的多帧图像，只输出有效JSON。",
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        }

        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"]).strip()

    def _evaluate_ollama(self, prompt: str, ordered_images: list[str], timeout: int) -> str:
        url = f"{self.ollama_base_url}/api/chat"
        images_b64 = [self._encode_image_base64(p) for p in ordered_images]

        body = {
            "model": self.ollama_model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "你是自动驾驶视频安全评测专家。你将收到按时间排序的多帧图像，只输出有效JSON。",
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                },
            ],
            "options": {
                "temperature": 0,
            },
        }

        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("message", {}).get("content", "")).strip()
