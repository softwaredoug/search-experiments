def pytest_runtest_logstart(nodeid, location):
    print(f"START {nodeid}", flush=True)
