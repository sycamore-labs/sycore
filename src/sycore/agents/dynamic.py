"""Dynamic agent instantiated from YAML definitions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sycore.agents.base import Agent, AgentContext, AgentTool
from sycore.agents.factory import AgentDefinition, AgentFactory

logger = logging.getLogger(__name__)


class DynamicAgent(Agent):
    """Agent dynamically configured from YAML definition.

    This agent reads its configuration from AgentDefinition YAML files:
    - System prompt from PromptLoader
    - Tools/skills from SkillRegistry
    - Model and provider from definition

    It supports generic tool handling via registered handlers.

    Example YAML definition (agents/planner.yaml):
        name: planner
        description: Plans presentation structure
        prompt: planner  # References prompts/planner.md
        skills:
          - save_plan
          - web_search
        model: claude-sonnet-4-5-20250929
        parameters:
          max_tokens: 8192
          temperature: 0.7

    Usage:
        # From YAML by name
        agent = DynamicAgent.from_yaml("planner", context, app_name="sydeck")
        result = agent.run_with_prompt("Create a plan for...")

        # With explicit definition
        factory = AgentFactory(app_name="sydeck")
        definition = factory.get("planner")
        agent = DynamicAgent(context, definition)
    """

    def __init__(
        self,
        context: AgentContext,
        definition: AgentDefinition,
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
        app_name: str = "sycore",
    ):
        """Initialize dynamic agent from definition.

        Args:
            context: Agent execution context.
            definition: Agent definition from YAML.
            tool_handlers: Map of tool name -> handler function.
            app_name: Application name for loader paths.
        """
        self.definition = definition
        self._tool_handlers = tool_handlers or {}
        self._app_name = app_name

        # Set agent metadata from definition
        self.name = definition.name
        self.description = definition.description

        # Load system prompt from PromptLoader
        self.system_prompt = self._load_prompt(definition.prompt_name)

        # Set skill names for tool loading
        self.skill_names = definition.skills

        # Override model and provider if specified in definition
        if definition.model:
            inferred_provider = definition.get_provider()
            # Need to pick the right API key based on provider
            # If provider changes, we need the appropriate API key from metadata
            api_key = context.api_key
            if inferred_provider != context.provider:
                # Try to get provider-specific API key from metadata
                api_key = context.metadata.get(f"{inferred_provider}_api_key", api_key)

            context = AgentContext(
                provider=inferred_provider,  # type: ignore[arg-type]
                api_key=api_key,
                model=definition.model,
                output_dir=context.output_dir,
                on_progress=context.on_progress,
                metadata=context.metadata,
            )

        # Store parameters for LLM calls
        self._parameters = definition.metadata.get("parameters", {})

        # Call parent init (sets up client, loads tools from skills)
        super().__init__(context)

    def _load_prompt(self, prompt_name: str) -> str:
        """Load system prompt from prompt loader.

        Args:
            prompt_name: Name of the prompt to load.

        Returns:
            Prompt content string.
        """
        try:
            from sycore.prompts.loader import PromptLoader

            loader = PromptLoader(app_name=self._app_name)
            prompt = loader.load(prompt_name)
            return prompt.content
        except FileNotFoundError:
            logger.warning(f"Prompt '{prompt_name}' not found, using default")
            return f"You are a {self.name} agent. {self.description}"
        except Exception as e:
            logger.warning(f"Failed to load prompt '{prompt_name}': {e}")
            return f"You are a {self.name} agent. {self.description}"

    def register_tool_handler(
        self,
        tool_name: str,
        handler: Callable[..., Any],
    ) -> None:
        """Register a handler for a specific tool.

        Args:
            tool_name: Name of the tool.
            handler: Function to handle tool calls.
        """
        self._tool_handlers[tool_name] = handler

    def _handle_tool_call(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Handle a tool call using registered handlers.

        Args:
            tool_name: Name of the tool called.
            args: Tool arguments.

        Returns:
            Tool result.
        """
        # Check registered handlers first
        if tool_name in self._tool_handlers:
            return self._tool_handlers[tool_name](**args)

        # Fall back to method-based handlers (tool_<name>)
        method_name = f"tool_{tool_name}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(**args)

        logger.warning(f"No handler for tool: {tool_name}")
        return {"error": f"No handler registered for tool: {tool_name}"}

    def run(self) -> dict[str, Any]:
        """Run the agent with the loaded configuration.

        Uses user_prompt from context.metadata if available.

        Returns:
            Agent execution result.
        """
        user_prompt = self.context.metadata.get("user_prompt", "")
        if not user_prompt:
            return {"error": "No user_prompt in context.metadata"}

        return self.run_with_prompt(user_prompt)

    def run_with_prompt(self, user_prompt: str) -> dict[str, Any]:
        """Run agent with a specific user prompt.

        Args:
            user_prompt: The user message to send.

        Returns:
            Agent execution result with 'response' and optionally parsed JSON.
        """
        response = self._call_llm([{"role": "user", "content": user_prompt}])

        # Try to parse as JSON
        result = self._extract_json(response)
        if result:
            return result

        return {"response": response}

    def run_conversation(
        self,
        messages: list[dict[str, str]],
        use_tools: bool = True,
    ) -> str:
        """Run a multi-turn conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            use_tools: Whether to enable tool use.

        Returns:
            LLM response text.
        """
        return self._call_llm(messages, use_tools=use_tools)

    @classmethod
    def from_yaml(
        cls,
        agent_name: str,
        context: AgentContext,
        app_name: str = "sycore",
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
    ) -> DynamicAgent:
        """Create agent from YAML definition by name.

        Args:
            agent_name: Name of the agent (matches YAML filename).
            context: Agent execution context.
            app_name: Application name for paths.
            tool_handlers: Optional tool handlers.

        Returns:
            Configured DynamicAgent instance.

        Raises:
            ValueError: If agent definition not found.
        """
        factory = AgentFactory(app_name=app_name)
        definition = factory.get(agent_name)

        if definition is None:
            raise ValueError(f"Agent definition not found: {agent_name}")

        return cls(context, definition, tool_handlers, app_name)

    @classmethod
    def from_yaml_with_defaults(
        cls,
        agent_name: str,
        context: AgentContext,
        app_name: str = "sycore",
        package_defaults_path: str | None = None,
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
    ) -> DynamicAgent:
        """Create agent from YAML with package defaults path.

        This is useful for domain packages that have their own default agents.

        Args:
            agent_name: Name of the agent.
            context: Agent execution context.
            app_name: Application name for paths.
            package_defaults_path: Path to package's defaults/agents directory.
            tool_handlers: Optional tool handlers.

        Returns:
            Configured DynamicAgent instance.

        Raises:
            ValueError: If agent definition not found.
        """
        from pathlib import Path

        factory = AgentFactory(app_name=app_name)

        if package_defaults_path:
            factory.add_search_path(Path(package_defaults_path), priority=-1)

        definition = factory.get(agent_name)

        if definition is None:
            raise ValueError(f"Agent definition not found: {agent_name}")

        return cls(context, definition, tool_handlers, app_name)
