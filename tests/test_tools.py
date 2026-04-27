import numpy as np

from exps import tools as tools_mod


class _FakeMinilm:
    def encode(self, text):
        if isinstance(text, list):
            return np.stack([self.encode(item) for item in text])
        if text == "alpha":
            return np.array([1.0, 0.0])
        if text == "beta":
            return np.array([0.99, 0.01])
        return np.array([0.0, 1.0])


def test_guard_disallow_similar_queries(monkeypatch):
    monkeypatch.setattr(tools_mod, "_minilm_model", lambda *_args, **_kwargs: _FakeMinilm())

    agent_state = {}
    params = {"tool_name": "search_bm25_guarded", "query": "alpha"}

    err = tools_mod.guard_disallow_similar_queries(params, agent_state, threshold=0.9)
    assert err is None

    params["query"] = "beta"
    err = tools_mod.guard_disallow_similar_queries(params, agent_state, threshold=0.9)
    assert isinstance(err, str)

    params["query"] = "gamma"
    err = tools_mod.guard_disallow_similar_queries(params, agent_state, threshold=0.9)
    assert err is None
