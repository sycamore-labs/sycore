"""Base skill definitions for sycore agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sycore.agents.base import AgentTool


@dataclass
class SkillDefinition:
    """Definition of a skill (tool) that an agent can use.

    Skills are reusable tool definitions that can be loaded from
    SKILL.md files or registered programmatically.

    Attributes:
        name: Unique skill name.
        description: Human-readable description.
        parameters: JSON Schema for the skill's parameters.
        version: Skill version.
        category: Category for grouping skills.
        metadata: Additional metadata from the skill file.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    version: str = "1.0"
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_agent_tool(self) -> "AgentTool":
        """Convert to AgentTool instance.

        Returns:
            AgentTool instance compatible with agent base class.
        """
        from sycore.agents.base import AgentTool

        return AgentTool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary with name, description, and parameters.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
