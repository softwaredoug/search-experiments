# AGENTS.md

This repo runs various search relevance experiments, particularly agentic and code generation experiments.

Details can be examined in the docs/ folder. Particularly [docs](docs/prd.md)


# Development Practices

## Testing practices

Read the testing expectations to orient yourself at [test docs](docs/tests.md). Particularly the importance of e2e tests, and what that dictates about the structure of the code.

Importantly pay attention to the pre-commit and pre-push hooks

- pre-commit will run cheaper checks, fix these when they fail
- pre-push will warn, but not fail, when CI failures exist. You should take these seriously and try to resolve CI issues before pushing. You'll receive logs to diagnose the failures.

## Dependencies

This project depends on the cheat-at-search library located here for datasets and utilities:
https://github.com/softwaredoug/cheat-at-search

Install cheat-at-search directly from git.

(In turn, this depends on the searcharray library, which you may also to be familiar with
https://github.com/softwaredoug/searcharray)
