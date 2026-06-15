import json
import logging
import os

import google.generativeai as genai
from src.utils.config import settings

logger = logging.getLogger(__name__)


class LLMClient:

    def __init__(
        self,
        model: str = settings.DEFAULT_LLM_MODEL
    ):
        api_key = settings.GEMINI_API_KEY

        if not api_key:
            raise ValueError(
                "LLM APi key is missing"
            )

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model)

    def generate_json(
        self,
        prompt: str
    ) -> dict | list:

        response = self.model.generate_content(
            prompt
        )

        text = response.text.strip()

        text = text.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

        try:
            return json.loads(text)

        except Exception as e:

            logger.error(
                f"LLM JSON parse failed: {e}"
            )

            raise