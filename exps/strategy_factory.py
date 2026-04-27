from __future__ import annotations

from pathlib import Path

from exps.strategy_config import StrategyConfig, load_strategy_config, resolve_strategy_class


def strategy_params_for_config(
    strategy_config: StrategyConfig, *, device: str | None = None
) -> dict:
    params = dict(strategy_config.params)
    params.pop("description", None)
    if device:
        if strategy_config.type == "agentic" and "embeddings_device" not in params:
            tool_names = params.get("search_tools")
            if tool_names is None or "minilm" in tool_names:
                params["embeddings_device"] = device
        if strategy_config.type == "embedding" and "device" not in params:
            params["device"] = device
    return params


def requires_bm25(strategy_type: str, params: dict) -> bool:
    if strategy_type == "bm25":
        return True
    if strategy_type == "embedding":
        return False
    if strategy_type == "agentic":
        return True
    return True


def load_strategy(
    strategy_path: str,
    *,
    device: str | None = None,
    base_path: str | None = None,
) -> tuple[StrategyConfig, dict, bool]:
    strategy_config = load_strategy_config(strategy_path, base_path=base_path)
    params = strategy_params_for_config(strategy_config, device=device)
    return strategy_config, params, requires_bm25(strategy_config.type, params)


def create_strategy(
    strategy_config: StrategyConfig,
    *,
    corpus,
    workers: int = 1,
    params: dict | None = None,
    device: str | None = None,
    dataset: str | None = None,
    trace_path: Path | None = None,
):
    if params is None:
        params = strategy_params_for_config(strategy_config, device=device)
    params = dict(params)
    if strategy_config.type == "agentic":
        if trace_path is None:
            raise ValueError("trace_path is required for agentic strategies.")
        params["trace_path"] = trace_path
    strategy_cls = resolve_strategy_class(strategy_config.type)
    if hasattr(strategy_cls, "build"):
        build_kwargs = {
            "corpus": corpus,
            "workers": workers,
            "device": device,
        }
        if strategy_config.type != "agentic":
            build_kwargs["dataset"] = dataset
        strategy = strategy_cls.build(params, **build_kwargs)
    else:
        strategy = strategy_cls(corpus, workers=workers, **params)
    return strategy, params
