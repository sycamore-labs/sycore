"""Generic LLM client for multi-provider support."""

from __future__ import annotations

import json
from typing import Any, Literal

LLMProvider = Literal["openai", "anthropic", "google", "ollama"]


class LLMClient:
    """Client for interacting with LLM providers.

    Provides a unified interface for generating text and JSON responses
    across multiple LLM providers (OpenAI, Anthropic, Google, Ollama).

    Example:
        client = LLMClient(provider="anthropic", api_key="...")
        response = client.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Write a haiku about coding.",
        )
    """

    def __init__(
        self,
        provider: LLMProvider,
        api_key: str,
        model: str | None = None,
    ):
        """Initialize the LLM client.

        Args:
            provider: LLM provider to use.
            api_key: API key for the provider.
            model: Model name (optional, uses provider default).
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._get_default_model()

    def _get_default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-5-20250929",
            "google": "gemini-2.0-flash",
            "ollama": "llama3.2",
        }
        return defaults.get(self.provider, "gpt-4o")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        """Generate text response from the LLM.

        Args:
            system_prompt: System instructions.
            user_prompt: User message.
            json_mode: If True, request JSON output format.

        Returns:
            Generated text response.
        """
        if self.provider in ("openai", "ollama"):
            if json_mode:
                return self._generate_openai_json(system_prompt, user_prompt)
            return self._generate_openai_text(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt, json_mode)
        elif self.provider == "google":
            return self._generate_google(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Generate and parse JSON response.

        Args:
            system_prompt: System instructions.
            user_prompt: User message.

        Returns:
            Parsed JSON as dictionary.
        """
        response = self.generate(system_prompt, user_prompt, json_mode=True)
        return self._parse_json_response(response)

    def _generate_openai_json(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using OpenAI API with JSON mode."""
        from openai import OpenAI

        if self.provider == "ollama":
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
        else:
            client = OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        return response.choices[0].message.content or "{}"

    def _generate_openai_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate plain text using OpenAI API (no JSON mode)."""
        from openai import OpenAI

        if self.provider == "ollama":
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
        else:
            client = OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content or ""

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        """Generate using Anthropic API."""
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)

        system = system_prompt
        if json_mode:
            system += "\n\nReturn ONLY valid JSON, no other text."

        response = client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )

        # Extract text from response
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        return text

    def _generate_google(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using Google Gemini API."""
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        # Combine system and user prompts
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}"
        response = model.generate_content(full_prompt)

        return response.text

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """Parse LLM response as raw JSON dict."""
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e
