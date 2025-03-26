# CLAUDE.md - Guidelines for Agent Code Assistance

## Project: Prompt Tuner

### Commands
- Run: `uv run main.py`
- Install: `uv pip install -e .`
- Create venv: `uv venv`
- Activate venv: `source .venv/bin/activate.fish` 
- Lint: `ruff check .`
- Format: `ruff format .`
- Typecheck: `mypy .`
- Test: `pytest`
- Single test: `pytest -xvs tests/test_file.py::test_name`

### Style Guidelines
- **Python**: Follow PEP 8 and use Python 3.12+ features
- **Imports**: Group standard lib, third-party, local; alphabetized
- **Formatting**: Use 4 spaces, 88 char line length
- **Types**: Use type annotations for all functions and class attributes
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error handling**: Use explicit exception types, avoid bare except
- **Documentation**: Docstrings in Google format for all public APIs