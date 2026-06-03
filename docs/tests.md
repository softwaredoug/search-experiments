# Repo tests

Tests in this repo are organized into three groups

- unit tests - run on a single module / class, with its dependencies mocked appropriately
- e2e tests - run on a full end to end task, but with API boundaries mocked
- integration tests - run on a full end to end task, with minimal mocking, and real API calls

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

## Use the gh tool to check integration / other expensive CI 

Installed on this machine is the github CLI - gh.

A precommit hook will check the last integration run to examine whether its behaviing as expected.
If not, you should work to resolve the issue, rerunning whatever failing test, and / or raising the issue to the human.
