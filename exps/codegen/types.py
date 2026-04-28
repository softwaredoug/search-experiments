from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CodeGenEvalConfig(BaseModel):
    num_training_queries: int = 200
    num_validation_queries: int = 100
    training_seed: int = 5678
    validation_seed: int = 1234
    eval_margin: float = 0.003


class CodeGenEditConfig(BaseModel):
    guards: list[dict[str, Any]] = Field(default_factory=list)


class CodeGenTrainConfig(BaseModel):
    edit: CodeGenEditConfig = Field(default_factory=CodeGenEditConfig)
    eval: CodeGenEvalConfig = Field(default_factory=CodeGenEvalConfig)
    model: str = "gpt-5-mini"
    rounds: int = 10
    system_prompt: str | None = None
    search_tools: list = Field(default_factory=lambda: ["bm25"])
    try_out_patch: bool = True
    start_with: str | None = None


class CodeGenRunConfig(BaseModel):
    top_k: int = 10


class CodeGenArtifact(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    reranker_path: Path
    code: str
    metadata: dict[str, Any]
    search_fn: callable
