"""Skill registry for loading and managing skills from SKILL.md files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import frontmatter
import yaml

from sycore.skills.base import SkillDefinition


class SkillRegistry:
    """Registry for loading and managing skills from SKILL.md files.

    Supports layered configuration with customizable search paths.
    """

    def __init__(
        self,
        search_paths: list[Path] | None = None,
        app_name: str = "sycore",
        project_dir: Path | None = None,
    ):
        """Initialize the skill registry.

        Args:
            search_paths: List of directories to search for skills.
                         If None, uses layered config paths based on app_name.
            app_name: Application name for default path construction.
            project_dir: Project directory for .sy/{app_name}/ lookup.
        """
        if search_paths is None:
            project_root = project_dir or Path.cwd()

            # Search in order: project, user (first match wins)
            search_paths = [
                project_root / ".sy" / app_name / "skills",  # Project overrides
                Path.home() / ".sy" / app_name / "skills",  # User overrides
            ]

        self._search_paths = search_paths
        self._skills: dict[str, SkillDefinition] = {}

    def add_search_path(self, path: Path, priority: int = 0) -> None:
        """Add a search path for skills.

        Args:
            path: Directory to search for skill definitions.
            priority: 0 = highest priority (searched first), -1 = append to end.
        """
        if priority == 0:
            self._search_paths.insert(0, path)
        else:
            self._search_paths.append(path)

    def discover(self) -> None:
        """Discover and load all skills from search paths."""
        self._skills.clear()
        for search_path in self._search_paths:
            if not search_path.exists():
                continue

            # Look for SKILL.md files in subdirectories
            for skill_dir in search_path.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        self._load_skill(skill_file)

    def _load_skill(self, skill_path: Path) -> None:
        """Load a skill from a SKILL.md file.

        Args:
            skill_path: Path to the SKILL.md file.
        """
        try:
            post = frontmatter.load(skill_path)

            # Extract skill definition from frontmatter
            name = post.get("name", skill_path.parent.name)

            # Don't override if already loaded (higher priority path wins)
            if name in self._skills:
                return

            description = post.get("description", "")
            version = post.get("version", "1.0")
            category = post.get("category", "general")

            # Parse parameters from content or frontmatter
            parameters = post.get("parameters", {})
            if not parameters and post.content:
                # Try to parse YAML/JSON from content
                parameters = self._parse_parameters_from_content(post.content)

            skill = SkillDefinition(
                name=name,
                description=description,
                parameters=parameters,
                version=version,
                category=category,
                metadata=dict(post.metadata),
            )

            self._skills[name] = skill

        except Exception:
            # Skip invalid skill files
            pass

    def _parse_parameters_from_content(self, content: str) -> dict[str, Any]:
        """Parse parameters from markdown content.

        Args:
            content: Markdown content that may contain YAML/JSON.

        Returns:
            Parsed parameters dict.
        """
        # Look for YAML code block
        if "```yaml" in content:
            start = content.find("```yaml") + 7
            end = content.find("```", start)
            if end > start:
                yaml_str = content[start:end].strip()
                try:
                    return yaml.safe_load(yaml_str)
                except yaml.YAMLError:
                    pass

        # Look for JSON code block
        if "```json" in content:
            import json

            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                json_str = content[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        return {}

    def get(self, name: str) -> SkillDefinition | None:
        """Get a skill by name.

        Args:
            name: Name of the skill.

        Returns:
            SkillDefinition or None if not found.
        """
        return self._skills.get(name)

    def list_names(self) -> list[str]:
        """List all registered skill names.

        Returns:
            List of skill names.
        """
        return sorted(self._skills.keys())

    def list_skills(self) -> list[SkillDefinition]:
        """List all registered skills.

        Returns:
            List of SkillDefinition instances.
        """
        return list(self._skills.values())

    def get_by_category(self, category: str) -> list[SkillDefinition]:
        """Get skills by category.

        Args:
            category: Category name.

        Returns:
            List of skills in the category.
        """
        return [s for s in self._skills.values() if s.category == category]

    def register(self, skill: SkillDefinition) -> None:
        """Register a skill manually.

        Args:
            skill: SkillDefinition to register.
        """
        self._skills[skill.name] = skill

    def clear(self) -> None:
        """Clear all registered skills."""
        self._skills.clear()


# Global registry instance
_registry: SkillRegistry | None = None


def get_skill_registry(app_name: str = "sycore") -> SkillRegistry:
    """Get the global skill registry instance.

    Args:
        app_name: Application name for path construction.

    Returns:
        SkillRegistry instance with discovered skills.
    """
    global _registry
    if _registry is None:
        _registry = SkillRegistry(app_name=app_name)
        _registry.discover()
    return _registry


def reset_skill_registry() -> None:
    """Reset the global skill registry (useful for testing)."""
    global _registry
    _registry = None
