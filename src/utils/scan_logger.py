import datetime
import os
import sys
import traceback
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


def install_global_crash_logger(log_dir: Path | str | None = None) -> None:
    """
    Installs a sys.excepthook crash handler that logs uncaught exceptions
    and full tracebacks to logs/crash.log without terminating the GUI thread.
    """
    target_dir = Path(log_dir) if log_dir else Path("logs")
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    crash_file = target_dir / "crash.log"

    def _crash_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        entry = f"[{ts}] UNCAUGHT EXCEPTION:\n{tb_text}\n{'=' * 70}\n"
        try:
            with open(crash_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            pass
        sys.stderr.write(entry)
        sys.stderr.flush()

    sys.excepthook = _crash_handler
