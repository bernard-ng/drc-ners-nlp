# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/drc_names_classifier/`, using a standard Python package layout.
- Project metadata and build configuration are in `pyproject.toml`.

## Build, Test, and Development Commands
- If you use uv (see `uv.lock`), `uv sync` installs dependencies and `uv run drc-names-classifier` runs the entry point.
- Run `uv run pyright` and `uv run ruff check .` to validate changes end-to-end.

## Coding Style & Naming Conventions
- Use Python 3.13 syntax and type hints when introducing new functions.
- Indentation: 4 spaces; line length: keep lines reasonably short (around 88–100 chars).
- Use `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Prefer small, focused functions;
