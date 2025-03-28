# PydanticAI Agent Framework

The PydanticAI Agent framework provides a robust way to use Pydantic with Large Language Models (LLMs). It offers a convenient interface for building AI agents that can execute complex workflows, use tools, and handle structured data validation.

## Overview

The Agent class is the main entry point for creating an AI agent that can:
- Run prompts against LLMs
- Define and execute tools
- Process and validate structured responses
- Handle conversation context and history
- Implement complex workflows

## Core Features

### Creating an Agent

```python
from pydantic_ai import Agent

# Basic agent
agent = Agent('openai:gpt-4o')

# Agent with dependencies type
agent = Agent('test', deps_type=str)
```

### System Prompts

System prompts provide instructions and context to the model. They can be defined as functions:

```python
@agent.system_prompt
def simple_system_prompt() -> str:
    return 'foobar'

@agent.system_prompt(dynamic=True)
async def async_system_prompt(ctx: RunContext[str]) -> str:
    return f'{ctx.deps} is the best'
```

### Tools

Tools allow the agent to perform actions and access external functionality:

```python
@agent.tool
def foobar(ctx: RunContext[int], x: int) -> int:
    return ctx.deps + x

@agent.tool(retries=2)
async def spam(ctx: RunContext[str], y: float) -> float:
    return ctx.deps + y
```

### Result Validation

Result validators can check and transform model responses:

```python
@agent.result_validator
def result_validator_simple(data: str) -> str:
    if 'wrong' in data:
        raise ModelRetry('wrong response')
    return data

@agent.result_validator
async def result_validator_deps(ctx: RunContext[str], data: str) -> str:
    if ctx.deps in data:
        raise ModelRetry('wrong response')
    return data
```

### Running the Agent

Several methods are available to run the agent:

```python
# Synchronous run
result = agent.run_sync('What is the capital of France?', deps='context')

# Asynchronous run
result = await agent.run('What is the capital of France?', deps='context')

# Streaming results with iteration
async with agent.iter('What is the capital of France?') as agent_run:
    async for node in agent_run:
        # Process each step in the execution
        print(node)
```

### Agent Run Results

The agent returns structured results:

```python
result = agent.run_sync('foobar', deps=1)
print(result.data)  # Access the final result data
print(result.messages)  # Access the conversation history
```

## Advanced Features

### Context Management

Override dependencies or models temporarily:

```python
with agent.override(deps='test_deps', model='openai:gpt-3.5-turbo'):
    result = agent.run_sync('prompt')
```

### Streaming Results

Get real-time results from the agent's execution:

```python
async def main():
    nodes = []
    async with agent.iter('What is the capital of France?') as agent_run:
        async for node in agent_run:
            nodes.append(node)
    print(nodes)
    print(agent_run.result.data)  # Paris
```

### Usage Limits

Control token usage and requests:

```python
from pydantic_ai import UsageLimits

result = agent.run_sync(
    'prompt',
    usage_limits=UsageLimits(max_requests=10, max_tokens=1000)
)
```

## Key Components

- **RunContext**: Provides context for tools and system prompts
- **ModelMessage**: Represents messages in the conversation history
- **AgentRun**: Represents an execution of the agent
- **ModelSettings**: Configures model behavior
- **UsageLimits**: Controls resource usage

## Error Handling

```python
from pydantic_ai import ModelRetry

@agent.result_validator
def validate_result(data: str) -> str:
    if not valid_check(data):
        raise ModelRetry("Invalid response, please try again")
    return data
```

## Integration with Pydantic

The framework leverages Pydantic for data validation, allowing you to define structured schemas for tool parameters and results.

```python
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

@agent.tool
def search(query: str) -> SearchResult:
    # Implementation...
    return SearchResult(title="...", url="...", snippet="...")
```

## Best Practices

1. Define clear system prompts to guide the model
2. Use typed dependencies for better IDE support
3. Implement result validators for robust responses
4. Use the async API for better performance in asynchronous applications
5. Leverage the tool decorator for external functionality

See [GitHub repository](https://github.com/pydantic/pydantic-ai) for the latest updates and documentation.