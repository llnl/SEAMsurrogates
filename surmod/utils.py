"""Utility functions for the surmod package."""

from pathlib import Path


def log_results(log_message: str, path_to_log: Path) -> None:
    """
    Append log message to file.

    Args:
        log_message: String to write to the log file.
        path_to_log: Path object pointing to the log file.
    """
    path_to_log.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_log, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")
