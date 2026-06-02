"""Agentic runner integration tests.

See docs/runner_tests_prd.md for requirements.
"""

import os

from exps.runners.run import RunParams, run_benchmark


def test_run_benchmark_agentic_hello_world():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    params = RunParams(
        strategy_path="configs/agentic_hello_world_bm25.yml",
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty
