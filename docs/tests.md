# Repo tests

Tests in this repo are organized into three groups

- unit tests - run on a single module / class, with its dependencies mocked appropriately
- e2e tests - run on a full end to end task, but with API boundaries mocked
- integration tests - run on a full end to end task, with minimal mocking, and real API calls

## What you *should* do for tests

The best tests to run are e2e. We mock anything expensive (like API calls) and run the full end to end task. We like this to match the functionality of the system, while focusing on what a real use case looks like.

Some cases call for robust unit tests - with clear interface boundaries, and specific / detailed algorithmic processing.

Integration tests are a last resort. If an integration test catches a problem, but no e2e catche it: that's a bug. And we should create an e2e test that catches it. Integration tests should be reserved for basic smoke testing, etc.

## How tests dictate design

Integration + e2e tests leverage the seperation between the CLI frontends and the python functions that implement those behaviors. There should be clean seperation where CLI-only parsing occurs in the CLI frontend. But most business logic occurs in the actual implementation. As an example, there is `run_benchmark` that implements the behavior of `uv run run` script.

That's the entry point for tests.

* Inline the yaml configuration of the experiment
* Run the experiment, via run_benchmark or other runner
* Check behaviors

## End-to-end tests

See [docs/e2e_tests.md](docs/e2e_tests.md) for details.

## Integration tests

See [docs/runner_tests_prd.md](docs/runner_tests_prd.md) for details.

## Precommit hooks

All e2e and unit tests are run on precommit. For this reason, they should be kept relatively lightweight. Integration tests are not run on precommit, and can be heavier.

## Prepush hooks

You receive a warning on integration test failures in a pre-push hook on CI failures.

The prepush hook will not fail, but you should take these seriously and try to fix CI issues.

You can rerun failed integration tests locally, and try to resolve the issue.
