from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
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
        self.max_images_per_request = int(os.getenv("LLM_MAX_IMAGES_PER_REQUEST", "12"))
        self.image_detail = os.getenv("LLM_IMAGE_DETAIL", "low")
        self.request_timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
        self.request_retries = int(os.getenv("LLM_REQUEST_RETRIES", "2"))

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

        if len(ordered) <= self.max_images:
            return ordered

        # 全序列均匀采样，避免只取前几帧导致漏检。
        sampled_indices = np.linspace(0, len(ordered) - 1, self.max_images).astype(int).tolist()
        sampled_indices = list(dict.fromkeys(sampled_indices))
        return [ordered[i] for i in sampled_indices]

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
                "camera_order": "每张图是单相机视图，文件名与图内文字包含 cam0..cam5",
            },
            "frame_groups": {
                "raw": len(keyframes.get("raw", [])),
                "det": len(keyframes.get("det", [])),
                "gen": len(keyframes.get("gen", [])),
            },
            "image_order": "按时间和相机顺序重复 [raw_(t,cam), det_(t,cam), gen_(t,cam)]",
            "rule_context": rule_context or {},
        }
        return self.template.replace("{{PAYLOAD_JSON}}", json.dumps(payload, ensure_ascii=False, indent=2))

    @staticmethod
    def _to_float01(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            out = default
        return float(max(0.0, min(1.0, out)))

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return False

    def _normalize_output(self, video_path: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        video_id = str(parsed.get("video_id") or Path(video_path).name)

        scores_raw = parsed.get("scores") or {}
        if not isinstance(scores_raw, dict):
            scores_raw = {}

        semantic = self._to_float01(scores_raw.get("semantic", parsed.get("semantic_score", 0.0)))
        logical = self._to_float01(scores_raw.get("logical", parsed.get("logical_score", 0.0)))
        decision = self._to_float01(scores_raw.get("decision", parsed.get("decision_score", 0.0)))

        final_score = self._to_float01(parsed.get("final_score", 0.4 * semantic + 0.3 * logical + 0.3 * decision))

        attacker_level = parsed.get("attacker_level", [])
        if isinstance(attacker_level, str):
            attacker_level = [x.strip() for x in attacker_level.replace("|", ",").split(",") if x.strip()]
        elif not isinstance(attacker_level, list):
            attacker_level = []

        normalized_levels: list[str] = []
        known = {"semantic": "Semantic", "logical": "Logical", "decision": "Decision"}
        for item in attacker_level:
            key = str(item).strip().lower()
            if key in known and known[key] not in normalized_levels:
                normalized_levels.append(known[key])

        if semantic >= 0.5 and "Semantic" not in normalized_levels:
            normalized_levels.append("Semantic")
        if logical >= 0.5 and "Logical" not in normalized_levels:
            normalized_levels.append("Logical")
        if decision >= 0.5 and "Decision" not in normalized_levels:
            normalized_levels.append("Decision")

        is_posioned = self._to_bool(parsed.get("is_posioned", final_score >= 0.5 or len(normalized_levels) > 0))
        reasoning = str(parsed.get("reasoning") or "No explicit risk reason provided by the model.")

        return {
            "video_id": video_id,
            "is_posioned": is_posioned,
            "attacker_level": normalized_levels,
            "scores": {
                "semantic": semantic,
                "logical": logical,
                "decision": decision,
            },
            "final_score": final_score,
            "reasoning": reasoning,
        }

    def evaluate(
        self,
        video_path: str,
        keyframes: Dict[str, list[str]],
        rule_context: Optional[Dict[str, object]] = None,
        timeout: int | None = None,
    ) -> Optional[Dict[str, object]]:
        if not self.is_enabled():
            return None

        ordered_images = self._ordered_image_paths(keyframes)
        if not ordered_images:
            return {
                "video_id": Path(video_path).name,
                "is_posioned": False,
                "attacker_level": [],
                "scores": {
                    "semantic": 0.0,
                    "logical": 0.0,
                    "decision": 0.0,
                },
                "final_score": 0.0,
                "reasoning": "No usable keyframes were found for evaluation.",
            }

        if timeout is None:
            timeout = self.request_timeout

        prompt = self._build_prompt(video_path, keyframes, rule_context=rule_context)

        batch_size = max(1, self.max_images_per_request)
        chunks = [ordered_images[i : i + batch_size] for i in range(0, len(ordered_images), batch_size)]
        normalized_parts: list[Dict[str, Any]] = []

        for chunk_idx, image_chunk in enumerate(chunks):
            chunk_prompt = prompt
            if len(chunks) > 1:
                chunk_prompt = (
                    prompt
                    + "\n\n补充要求: 当前仅提供了该视频的一部分图像。"
                    + f"这是第 {chunk_idx + 1}/{len(chunks)} 个图像分块，请仅基于本分块打分。"
                )

            if self.backend in {"api", "openai_compatible"}:
                text = self._evaluate_openai_compatible(chunk_prompt, image_chunk, timeout)
            elif self.backend == "ollama":
                text = self._evaluate_ollama(chunk_prompt, image_chunk, timeout)
            else:
                raise RuntimeError(f"不支持的 LLM_BACKEND: {self.backend}")

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    parsed = json.loads(text[start : end + 1])
                else:
                    raise

            normalized_parts.append(self._normalize_output(video_path=video_path, parsed=parsed))

        if len(normalized_parts) == 1:
            return normalized_parts[0]

        # 多分块聚合: 取各层最高风险，保证不会因局部遗漏压低总分。
        sem = max(float(x["scores"]["semantic"]) for x in normalized_parts)
        log = max(float(x["scores"]["logical"]) for x in normalized_parts)
        dec = max(float(x["scores"]["decision"]) for x in normalized_parts)
        final_score = max(float(x["final_score"]) for x in normalized_parts)

        levels: list[str] = []
        for x in normalized_parts:
            for level in x.get("attacker_level", []):
                if isinstance(level, str) and level not in levels:
                    levels.append(level)

        reasoning_src = max(normalized_parts, key=lambda x: float(x.get("final_score", 0.0)))
        reasoning = str(reasoning_src.get("reasoning", "No explicit risk reason provided by the model."))

        return {
            "video_id": Path(video_path).name,
            "is_posioned": bool(final_score >= 0.5 or len(levels) > 0),
            "attacker_level": levels,
            "scores": {
                "semantic": sem,
                "logical": log,
                "decision": dec,
            },
            "final_score": float(max(0.0, min(1.0, final_score))),
            "reasoning": reasoning,
        }

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

        last_error: Exception | None = None
        for _ in range(max(1, self.request_retries + 1)):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                return str(data["choices"][0]["message"]["content"]).strip()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                last_error = e
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenAI-compatible evaluation failed without explicit exception.")

    def _evaluate_ollama(self, prompt: str, ordered_images: list[str], timeout: int) -> str:
        images_b64 = [self._encode_image_base64(p) for p in ordered_images]
        headers = {"Content-Type": "application/json"}

        chat_url = f"{self.ollama_base_url}/api/chat"
        chat_body = {
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

        try:
            resp = requests.post(chat_url, headers=headers, json=chat_body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("message", {}).get("content", "")).strip()
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code != 404:
                raise

        # 兼容部分部署不支持 /api/chat 的场景，降级到 /api/generate。
        generate_url = f"{self.ollama_base_url}/api/generate"
        generate_body = {
            "model": self.ollama_model,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }

        resp = requests.post(generate_url, headers=headers, json=generate_body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", "")).strip()
