from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Rounds:
    rounds_log_path: Path
    messages: list[str] = field(default_factory=list)
    round_summaries: list[dict] = field(default_factory=list)
    round_ndcgs: list[float] = field(default_factory=list)
    round_test_ndcgs: list[float] = field(default_factory=list)

    def finish(
        self,
        *,
        round_number: int,
        short_name: str | None,
        summary: str | None,
        message: str | None,
        mean_ndcg: float,
        mean_test_ndcg: float,
        training_query_count: int,
        validation_query_count: int,
        test_query_count: int,
    ) -> dict:
        round_record = {
            "round": round_number,
            "short_name": short_name,
            "summary": summary,
            "message": message,
            "mean_ndcg": mean_ndcg,
            "mean_ndcg_test": mean_test_ndcg,
            "training_query_count": training_query_count,
            "validation_query_count": validation_query_count,
            "test_query_count": test_query_count,
        }
        self.round_summaries.append(round_record)
        self.round_ndcgs.append(mean_ndcg)
        self.round_test_ndcgs.append(mean_test_ndcg)
        if message:
            self.messages.append(message)
        payload = json.dumps(round_record)
        with self.rounds_log_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        return round_record
