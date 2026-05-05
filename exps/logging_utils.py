from __future__ import annotations

import logging
from pathlib import Path


def log_to_stdout(logger_name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_to_path_and_stdout(
    logger_name: str,
    log_path: str | Path,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = log_to_stdout(logger_name, level=level)
    log_to_file(logger, log_path, level=level)
    return logger


def log_to_file(
    logger: logging.Logger,
    log_path: str | Path,
    level: int | None = None,
) -> None:
    path = Path(log_path).expanduser().resolve()
    handler_exists = False
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(path):
            handler_exists = True
            break
    if handler_exists:
        return
    handler = logging.FileHandler(path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    if level is not None:
        handler.setLevel(level)
    logger.addHandler(handler)
