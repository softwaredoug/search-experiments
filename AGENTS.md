# AGENTS.md

This repo runs various search relevance experiments, particularly agentic and code generation experiments.

Details can be examined in the docs/ folder. Particularly [docs](docs/prd.md)

## General project notes

This is a python project. Managed by uv.

## Testing notes

Read the testing expectations to orient yourself at [test docs](docs/tests.md)

Importantly pay attention to the pre-commit and pre-push hooks

- pre-commit will run cheaper checks, fix these when they fail
- pre-push will warn, but not fail, when CI failures exist. You should take these seriously and try to resolve CI issues before pushing. You'll receive logs to diagnose the failures.
