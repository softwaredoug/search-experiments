from concurrent.futures import ThreadPoolExecutor

from exps.resource_tracking import record_model_call, track_model_calls


def test_model_call_counters_are_isolated_between_search_threads():
    def count_calls(count: int) -> int:
        with track_model_calls() as calls:
            for _ in range(count):
                record_model_call()
        return calls.calls

    with ThreadPoolExecutor(max_workers=2) as executor:
        counts = list(executor.map(count_calls, (2, 5)))

    assert counts == [2, 5]
    record_model_call()  # Calls outside a tracked search are ignored.
