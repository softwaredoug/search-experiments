from __future__ import annotations

import random


def split_queries(
    *,
    base_queries: list[str],
    train_size: int,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if not base_queries:
        return [], [], []
    if train_size >= len(base_queries):
        return list(base_queries), [], []
    shuffled = list(base_queries)
    random.Random(seed).shuffle(shuffled)
    training_queries = shuffled[:train_size]
    validation_queries = shuffled[train_size:]
    return training_queries, validation_queries, list(validation_queries)


def build_round_queries(
    *,
    round_idx: int,
    base_queries: list[str],
    train_size: int,
    eval_cfg,
) -> tuple[list[str], list[str], list[str]]:
    training_seed = eval_cfg.seed + round_idx
    training_queries_list, validation_queries_list, test_queries_list = split_queries(
        base_queries=base_queries,
        train_size=train_size,
        seed=training_seed,
    )
    return training_queries_list, validation_queries_list, test_queries_list
