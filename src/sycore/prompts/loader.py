"""Prompt loader for externalized prompts with Jinja2 templating support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
from jinja2 import Template


@dataclass
class Prompt:
    """A loaded prompt with metadata and content.

    Prompts are stored as markdown files with YAML frontmatter.
    They support Jinja2 templating for dynamic content.
    """

    name: str
    description: str
    version: str
    content: str
    metadata: dict[str, Any]

    def render(self, context: dict[str, Any] | None = None) -> str:
        """Render the prompt with Jinja2 templating.

        Args:
            context: Variables to substitute in the template.

        Returns:
            Rendered prompt string.
        """
        if context is None:
            context = {}
        template = Template(self.content)
        return template.render(**context)


class PromptLoader:
    """Loads prompts from external markdown files with YAML frontmatter.

    Supports layered configuration with customizable search paths.
    """

    def __init__(
        self,
        search_paths: list[Path] | None = None,
        app_name: str = "sycore",
        project_dir: Path | None = None,
    ):
        """Initialize the prompt loader.

        Args:
            search_paths: List of directories to search for prompts.
                         If None, uses layered config paths based on app_name.
            app_name: Application name for default path construction.
            project_dir: Project directory for .sy/{app_name}/ lookup.
        """
        if search_paths is None:
            project_root = project_dir or Path.cwd()

            # Search in order: project, user (first match wins)
            search_paths = [
                project_root / ".sy" / app_name / "prompts",  # Project overrides
                Path.home() / ".sy" / app_name / "prompts",  # User overrides
            ]

        self._search_paths = search_paths
        self._cache: dict[str, Prompt] = {}

    def add_search_path(self, path: Path, priority: int = 0) -> None:
        """Add a search path for prompts.

        Args:
            path: Directory to search for prompt files.
            priority: 0 = highest priority (searched first), -1 = append to end.
        """
        if priority == 0:
            self._search_paths.insert(0, path)
        else:
            self._search_paths.append(path)

    def _find_prompt_file(self, name: str) -> Path | None:
        """Find a prompt file by name in search paths.

        Args:
            name: Name of the prompt (without extension).

        Returns:
            Path to the prompt file, or None if not found.
        """
        for search_path in self._search_paths:
            prompt_path = search_path / f"{name}.md"
            if prompt_path.exists():
                return prompt_path
        return None

    def load(self, name: str) -> Prompt:
        """Load a prompt by name.

        Args:
            name: Name of the prompt (e.g., 'planner', 'critic').

        Returns:
            Loaded Prompt object.

        Raises:
            FileNotFoundError: If prompt file doesn't exist.
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        prompt_path = self._find_prompt_file(name)
        if prompt_path is None:
            raise FileNotFoundError(
                f"Prompt '{name}' not found in search paths: {self._search_paths}"
            )

        # Parse frontmatter and content
        post = frontmatter.load(prompt_path)

        prompt = Prompt(
            name=post.get("name", name),
            description=post.get("description", ""),
            version=post.get("version", "1.0"),
            content=post.content,
            metadata=dict(post.metadata),
        )

        # Cache and return
        self._cache[name] = prompt
        return prompt

    def render(self, name: str, context: dict[str, Any] | None = None) -> str:
        """Load and render a prompt with context.

        Args:
            name: Name of the prompt.
            context: Variables to substitute in the template.

        Returns:
            Rendered prompt string.
        """
        prompt = self.load(name)
        return prompt.render(context)

    def list_prompts(self) -> list[tuple[str, str]]:
        """List all available prompts.

        Returns:
            List of (name, source) tuples.
        """
        prompts = []
        for search_path in self._search_paths:
            if search_path.exists():
                source = "project" if ".sy" in str(search_path) else "default"
                for prompt_file in search_path.glob("*.md"):
                    name = prompt_file.stem
                    prompts.append((name, source))
        return prompts

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()


# Global loader instance
_loader: PromptLoader | None = None


def get_prompt_loader(app_name: str = "sycore") -> PromptLoader:
    """Get the global prompt loader instance.

    Args:
        app_name: Application name for path construction.

    Returns:
        PromptLoader instance.
    """
    global _loader
    if _loader is None:
        _loader = PromptLoader(app_name=app_name)
    return _loader


def load_prompt(name: str) -> Prompt:
    """Convenience function to load a prompt."""
    return get_prompt_loader().load(name)


def render_prompt(name: str, context: dict[str, Any] | None = None) -> str:
    """Convenience function to load and render a prompt."""
    return get_prompt_loader().render(name, context)


def reset_prompt_loader() -> None:
    """Reset the global prompt loader (useful for testing)."""
    global _loader
    _loader = None
