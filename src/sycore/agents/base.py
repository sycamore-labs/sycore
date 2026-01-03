"""Base agent class for sycore AI agents."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from sycore.skills.base import SkillDefinition


@dataclass
class AgentContext:
    """Generic agent execution context.

    This provides the base context for agent execution. Domain-specific
    implementations should extend this class to add their own fields.

    Attributes:
        provider: LLM provider to use.
        api_key: API key for the provider.
        model: Model name to use.
        output_dir: Directory for output files.
        on_progress: Callback for progress reporting.
        metadata: Generic extension point for domain-specific data.
    """

    provider: Literal["anthropic", "openai", "google", "ollama"]
    api_key: str
    model: str
    output_dir: str = ""
    on_progress: Callable[[str], None] | None = None

    # Generic shared state - extensions add domain-specific fields
    metadata: dict[str, Any] = field(default_factory=dict)

    def report_progress(self, message: str) -> None:
        """Report progress to caller."""
        if self.on_progress:
            self.on_progress(message)


class AgentTool(BaseModel):
    """Definition of a tool an agent can use."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_anthropic(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Agent(ABC):
    """Base class for sycore AI agents.

    Agents are autonomous units that can interact with LLMs and use tools
    to accomplish tasks. This base class provides:

    - Multi-provider LLM support (Anthropic, OpenAI, Google, Ollama)
    - Tool/function calling with automatic handling
    - Skill-based tool loading from registry
    - JSON extraction utilities

    Subclasses must implement the `run()` method.
    """

    name: str = "base"
    description: str = "Base agent"
    system_prompt: str = ""
    tools: list[AgentTool] = []
    # Skill names to load from registry (alternative to inline tools)
    skill_names: list[str] = []

    def __init__(self, context: AgentContext):
        self.context = context
        self._client = None
        # Load tools from skills if skill_names are specified
        if self.skill_names and not self.tools:
            self.tools = self._load_tools_from_skills()

    def _load_tools_from_skills(self) -> list[AgentTool]:
        """Load tools from the skills registry based on skill_names.

        Returns:
            List of AgentTool instances loaded from skills.
        """
        from sycore.skills import get_skill_registry

        registry = get_skill_registry()
        tools = []

        for skill_name in self.skill_names:
            skill = registry.get(skill_name)
            if skill:
                tools.append(skill.to_agent_tool())

        return tools

    @classmethod
    def get_skills(cls) -> list[SkillDefinition]:
        """Get skill definitions for this agent.

        Returns:
            List of SkillDefinition instances.
        """
        from sycore.skills import get_skill_registry

        registry = get_skill_registry()
        return [
            skill
            for name in cls.skill_names
            if (skill := registry.get(name)) is not None
        ]

    @property
    def client(self) -> Any:
        """Get or create LLM client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        """Create LLM client based on provider."""
        if self.context.provider == "anthropic":
            from anthropic import Anthropic

            return Anthropic(api_key=self.context.api_key)
        elif self.context.provider == "openai":
            from openai import OpenAI

            return OpenAI(api_key=self.context.api_key)
        elif self.context.provider == "ollama":
            from openai import OpenAI

            return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        elif self.context.provider == "google":
            import google.generativeai as genai

            genai.configure(api_key=self.context.api_key)
            return genai.GenerativeModel(self.context.model)
        else:
            raise ValueError(f"Unsupported provider: {self.context.provider}")

    def _call_llm(self, messages: list[dict[str, str]], use_tools: bool = True) -> str:
        """Call the LLM and handle tool use."""
        if self.context.provider == "anthropic":
            return self._call_anthropic(messages, use_tools)
        elif self.context.provider in ("openai", "ollama"):
            return self._call_openai(messages, use_tools)
        elif self.context.provider == "google":
            return self._call_google(messages, use_tools)
        else:
            raise ValueError(f"Unsupported provider: {self.context.provider}")

    def _call_anthropic(self, messages: list[dict[str, str]], use_tools: bool) -> str:
        """Call Anthropic API."""
        # Get max_tokens from parameters if available
        max_tokens = getattr(self, '_parameters', {}).get('max_tokens', 4096)
        logger.debug(f"_call_anthropic: max_tokens={max_tokens}, model={self.context.model}")
        kwargs: dict[str, Any] = {
            "model": self.context.model,
            "max_tokens": max_tokens,
            "system": self.system_prompt,
            "messages": messages,
        }

        if use_tools and self.tools:
            kwargs["tools"] = [t.to_anthropic() for t in self.tools]

        response = self.client.messages.create(**kwargs)

        # Handle tool use
        all_messages: list[dict[str, Any]] = list(messages)
        while response.stop_reason == "tool_use":
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._handle_tool_call(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

            all_messages = all_messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]

            response = self.client.messages.create(**kwargs | {"messages": all_messages})

        # Log response info
        logger.debug(
            f"Anthropic response: stop_reason={response.stop_reason}, "
            f"output_tokens={response.usage.output_tokens}"
        )

        # Extract text response
        for block in response.content:
            if hasattr(block, "text"):
                return block.text

        # Log if no text found
        logger.warning(
            f"No text block in Anthropic response. "
            f"Stop reason: {response.stop_reason}, "
            f"Content types: {[type(b).__name__ for b in response.content]}"
        )
        return ""

    def _call_openai(self, messages: list[dict[str, str]], use_tools: bool) -> str:
        """Call OpenAI API."""
        kwargs: dict[str, Any] = {
            "model": self.context.model,
            "messages": [{"role": "system", "content": self.system_prompt}] + messages,
        }

        if use_tools and self.tools:
            kwargs["tools"] = [t.to_openai() for t in self.tools]

        response = self.client.chat.completions.create(**kwargs)

        # Handle tool calls
        while response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            messages.append(response.choices[0].message)

            for tool_call in tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = self._handle_tool_call(tool_call.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

            response = self.client.chat.completions.create(**kwargs | {"messages": messages})

        return response.choices[0].message.content or ""

    def _call_google(self, messages: list[dict[str, str]], use_tools: bool) -> str:
        """Call Google Gemini API."""
        # Combine system prompt with user messages
        prompt_parts = [self.system_prompt]
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {msg['content']}")

        response = self.client.generate_content("\n\n".join(prompt_parts))
        return response.text

    def _handle_tool_call(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Handle a tool call. Override in subclasses."""
        method_name = f"tool_{tool_name}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(**args)
        return {"error": f"Unknown tool: {tool_name}"}

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Run the agent and return results."""
        pass

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to find JSON block in markdown
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        # Try to parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

        return {}
