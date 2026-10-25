import datetime
import os
from pathlib import Path


def log_scan_error(worker_name: str, file_path: str | Path, error: Exception | str) -> None:
    """
    Appends broken or corrupted media file details to logs/scan_errors.log.
    Safe against I/O exceptions and creates the logs directory if it does not exist.
    """
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "scan_errors.log"
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [{worker_name}] Error processing '{file_path}': {error}\n")
    except Exception:
        pass
