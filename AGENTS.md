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

## General design principles

- Centrality of the "Strategy" class - many ways of implementing search are implemented as a SearchStrategy from (see cheat-at-search)
- Strategies are configured - ie in yml files in config. That's how we paramaterize them to run experiments
- Strategies are run with `uv run run ...` runner script
- Scripts in scripts/ run multiple strategies for some specific experiment (there's usually a corresponding python script to generate graphs and such)
- We store graphs in assetts
- There are writeups of some of these in research/
- An ability to turn a config yml into a notebook (see [docs/notebooks.md](docs/notebooks_prd.md))

### Runner design approach

The different runners (primarilly `uv run run` but others too) are designed to be e2e agent testable as possible

* A thin frontend script, ie (exps/runner.py) that uses argparse to process CLI into a data object, then call the right
  backend
* A thicker runners backend script, ie exps/runners/run.py that does the actual work (e2e tests test here)
