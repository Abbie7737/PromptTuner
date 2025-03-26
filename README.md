# Prompt Tuner

A tool that optimizes LLM prompts for small language models using a large language model.

## Overview

Prompt Tuner takes a base prompt intended for a small LLM (<32B parameters) and uses a large LLM to rewrite the prompt for better performance. The process follows these steps:

1. Generate multiple prompt variations using the large LLM
2. Evaluate each variation by running it on the small LLM and grading the response
3. Refine the top-performing prompts to further improve results
4. Select the best optimized prompt based on final evaluations
5. Generate a detailed report explaining the optimization process

## Features

- Uses PydanticAI as the agent framework
- Configurable large and small LLMs via LMStudio
- Extensible prompt templates stored in separate files
- Comprehensive evaluation metrics for prompt quality
- Detailed explanation of why the optimized prompt works better
- Complete process documentation in markdown reports

## Installation

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate.fish

# Install the package
uv pip install -e .
```

## Configuration

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

Edit the `.env` file to configure:
- LMStudio URL and model names
- Number of initial prompts to generate
- Results directory location

## Usage

```bash
# Optimize a prompt provided as a command line argument
uv run main.py "Write a poem about a cat"

# Optimize a prompt from a file with a task description
uv run main.py -f my_prompt.txt -t "Generate creative content about animals"

# Specify a custom environment file
uv run main.py "Write a poem about a cat" -e custom.env
```

## Output

The tool generates:
- A markdown report with the full optimization process
- A JSON file with raw result data
- Console output showing the best prompt and explanation

Results are saved in the configured results directory (default: `./results`).

## Project Structure

- `prompt_tuner/` - Main package
  - `core/` - Core functionality
    - `tuner.py` - Main prompt tuning implementation
  - `utils/` - Utility modules
    - `lmstudio.py` - LMStudio client wrapper
    - `prompt_loader.py` - Prompt template loader
  - `prompts/` - Prompt templates
    - `generator.prompt` - For generating variations
    - `evaluator.prompt` - For evaluating responses
    - `refiner.prompt` - For refining prompts
    - `explainer.prompt` - For explaining results
    - `report.prompt` - For generating reports
- `main.py` - CLI entry point