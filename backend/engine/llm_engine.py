import os
import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from .base import BaseEngine
from .mock_engine import MockEngine

logger = logging.getLogger(__name__)

class LLMEngine(BaseEngine):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        fallback_engine: Optional[BaseEngine] = None
    ):
        self.api_key = api_key or os.environ.get("AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("AI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("AI_MODEL") or "gpt-4o-mini"
        self.fallback_engine = fallback_engine or MockEngine()

    def update_config(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        if api_key is not None:
            self.api_key = api_key
        if base_url is not None:
            self.base_url = base_url.rstrip("/")
        if model is not None:
            self.model = model

    def generate_response(
        self,
        user_text: str,
        state_result: Dict[str, Any],
        history: List[Dict[str, str]]
    ) -> str:
        # If no API key is configured, immediately use MockEngine
        if not self.api_key:
            return self.fallback_engine.generate_response(user_text, state_result, history)

        context = state_result.get("context", {})
        system_prompt = context.get("system_prompt", "")

        messages = [{"role": "system", "content": system_prompt}]
        for h in history[-8:]:  # keep recent conversation turns
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        messages.append({"role": "user", "content": user_text})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 500
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
                logger.warning(f"LLM API returned status {resp.status_code}: {resp.text}. Falling back to mock engine.")
        except Exception as e:
            logger.warning(f"LLM API call failed with exception: {e}. Falling back to mock engine.")

        return self.fallback_engine.generate_response(user_text, state_result, history)
