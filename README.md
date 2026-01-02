# sycore

Generic agent/pipeline execution framework - the core of the Sycamore AI operating system driver.

## Overview

sycore provides a domain-agnostic framework for building AI agent pipelines. It is designed to be the foundational layer for autonomous AI systems that can be dynamically configured and extended.

### Key Features

- **Multi-Provider LLM Support**: Anthropic, OpenAI, Google Gemini, and Ollama
- **Pipeline Orchestration**: Composable, multi-stage workflows with YAML or programmatic configuration
- **Skills System**: Reusable tool definitions loaded from SKILL.md files
- **Prompt Management**: Externalized prompts with Jinja2 templating and YAML frontmatter
- **Layered Configuration**: Project > User > Package override hierarchy

## Installation

```bash
pip install sycore
```

## Architecture

```
sycore/
├── agents/          # Agent base classes and factory
│   ├── base.py      # Agent, AgentContext, AgentTool
│   └── factory.py   # AgentFactory, AgentDefinition
├── pipeline/        # Pipeline orchestration
│   ├── executor.py  # PipelineExecutor (domain-agnostic)
│   ├── stages.py    # PipelineStage ABC
│   ├── context.py   # BasePipelineContext, BasePipelineConfig
│   └── loader.py    # YAML pipeline definitions
├── skills/          # Tool/skill system
│   ├── base.py      # SkillDefinition
│   └── registry.py  # SkillRegistry
├── prompts/         # Prompt management
│   └── loader.py    # PromptLoader with Jinja2
├── resources/       # Configuration loading
│   └── loader.py    # ResourceLoader for YAML configs
└── core/            # Core utilities
    └── llm.py       # LLMClient wrapper
```

## Core Components

### 1. Agents

Agents are autonomous units that interact with LLMs and use tools to accomplish tasks.

```python
from sycore.agents import Agent, AgentContext, AgentTool

class MyAgent(Agent):
    name = "my-agent"
    description = "Does something useful"
    system_prompt = "You are a helpful assistant."

    # Define tools inline
    tools = [
        AgentTool(
            name="search",
            description="Search for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )
    ]

    # Or load tools from skills registry
    skill_names = ["search", "calculator"]

    def run(self) -> dict:
        response = self._call_llm([
            {"role": "user", "content": "Hello!"}
        ])
        return {"response": response}

    def tool_search(self, query: str) -> dict:
        """Handle the search tool call."""
        return {"results": ["result1", "result2"]}

# Create context and run
context = AgentContext(
    provider="anthropic",  # or "openai", "google", "ollama"
    api_key="...",
    model="claude-sonnet-4-5-20250929",
    metadata={"custom_field": "value"},  # Extension point
)
agent = MyAgent(context)
result = agent.run()
```

**Extending AgentContext:**

```python
from dataclasses import dataclass, field
from sycore.agents import AgentContext

@dataclass
class MyAppContext(AgentContext):
    """Domain-specific agent context."""
    template_name: str = ""
    custom_config: dict = field(default_factory=dict)
```

### 2. Pipeline Execution

Pipelines orchestrate multi-stage workflows. Unlike domain-specific executors, sycore's `PipelineExecutor` has NO hardcoded stages - all stages must be provided.

```python
from sycore.pipeline import PipelineExecutor, PipelineStage
from sycore.pipeline.context import BasePipelineContext, BasePipelineConfig, PipelineLogger

# Define a custom stage
class ProcessingStage(PipelineStage):
    name = "Processing"
    is_critical = True  # Pipeline stops on failure

    def should_run(self) -> bool:
        return True  # Conditional execution

    def run(self) -> None:
        self.logger.info("Processing data...")
        # Access shared state
        data = self.context.source_content
        # Update context with results
        self.context.output_path = Path("output.txt")

# Create context and config
context = BasePipelineContext(source_content="input data")
config = BasePipelineConfig(
    primary_provider="anthropic",
    primary_api_key="...",
)

# Option 1: Explicit stages
executor = PipelineExecutor(context, config)
executor._set_stages([
    ProcessingStage(context, config, PipelineLogger()),
])
output_path = executor.execute()

# Option 2: YAML-defined pipeline
executor = PipelineExecutor(
    context, config,
    pipeline_name="my_pipeline",
    stage_registry={"ProcessingStage": ProcessingStage},
)
```

**Extending PipelineContext:**

```python
from dataclasses import dataclass, field
from sycore.pipeline.context import BasePipelineContext

@dataclass
class MyPipelineContext(BasePipelineContext):
    """Domain-specific pipeline context."""
    processed_items: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
```

### 3. Skills System

Skills are reusable tool definitions that can be loaded from SKILL.md files or registered programmatically.

```python
from sycore.skills import SkillDefinition, get_skill_registry

# Define a skill
skill = SkillDefinition(
    name="web_search",
    description="Search the web for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 10}
        },
        "required": ["query"]
    },
    category="search",
)

# Register and use
registry = get_skill_registry()
registry.register(skill)

# Convert to provider format
anthropic_tool = skill.to_anthropic()
openai_tool = skill.to_openai()
agent_tool = skill.to_agent_tool()
```

### 4. Prompt Management

Externalized prompts with Jinja2 templating support.

**Prompt file format** (`prompts/planner.md`):
```markdown
---
name: planner
description: Plans the execution strategy
version: "1.0"
---
You are a planning assistant.

Given the following input:
{{ input_content }}

Create a plan with {{ num_steps }} steps.
```

**Usage:**
```python
from sycore.prompts import PromptLoader, render_prompt

# Using the loader directly
loader = PromptLoader(app_name="myapp")
prompt = loader.load("planner")
rendered = prompt.render({
    "input_content": "Build a website",
    "num_steps": 5
})

# Convenience function
rendered = render_prompt("planner", {"input_content": "...", "num_steps": 5})
```

### 5. Resource Loading

Generic YAML configuration loader with layered overrides.

```python
from sycore.resources import ResourceLoader, get_resource_loader

# Load themes
loader = get_resource_loader(resource_type="themes", app_name="myapp")
dark_theme = loader.load("dark")

# Load with defaults
theme = loader.load_with_defaults("dark", {
    "background": "#ffffff",
    "foreground": "#000000"
})

# List available resources
available = loader.list_resources()  # ["dark", "light", "solarized"]
```

## Configuration Paths

sycore uses a layered configuration system:

1. **Project level**: `./.sy/{app_name}/` (highest priority)
2. **User level**: `~/.sy/{app_name}/`
3. **Package defaults**: Built into the package (lowest priority)

Example directory structure:
```
my-project/
├── .sy/
│   └── myapp/
│       ├── prompts/
│       │   └── planner.md      # Override default planner
│       ├── resources/
│       │   └── custom.yaml     # Custom config
│       └── pipelines/
│           └── my_pipeline.yaml
```

## Creating a Domain-Specific Package

sycore is designed to be extended. Here's how to build on top of it:

```python
# myapp/pipeline/executor.py
from sycore.pipeline import PipelineExecutor
from myapp.pipeline.stages import Stage1, Stage2, Stage3

class MyAppExecutor(PipelineExecutor):
    """Domain-specific pipeline executor."""

    def __init__(self, context, config, **kwargs):
        super().__init__(
            context=context,
            config=config,
            stage_registry=self._get_stage_registry(),
            **kwargs,
        )

    @staticmethod
    def _get_stage_registry():
        return {
            "Stage1": Stage1,
            "Stage2": Stage2,
            "Stage3": Stage3,
        }
```

See [sydeck](https://github.com/sycamore-labs/sydeck) for a complete example of a domain-specific package built on sycore.

## LLM Provider Support

| Provider | Status | Notes |
|----------|--------|-------|
| Anthropic | Full | Claude models with tool use |
| OpenAI | Full | GPT models with function calling |
| Google | Basic | Gemini models (no tool use yet) |
| Ollama | Full | Local models via OpenAI-compatible API |

```python
# Anthropic
context = AgentContext(provider="anthropic", api_key="...", model="claude-sonnet-4-5-20250929")

# OpenAI
context = AgentContext(provider="openai", api_key="...", model="gpt-4o")

# Ollama (local)
context = AgentContext(provider="ollama", api_key="", model="llama3.2")

# Google Gemini
context = AgentContext(provider="google", api_key="...", model="gemini-1.5-pro")
```

## Development

```bash
# Clone and install
git clone https://github.com/sycamore-labs/sycore.git
cd sycore
uv sync --group dev

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## License

MIT
