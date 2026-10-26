import datetime
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from src.utils.paths import get_app_data_dir


def _get_log_dir() -> Path:
    """Returns a writable directory for application logs."""
    local_logs = Path("logs")
    try:
        local_logs.mkdir(parents=True, exist_ok=True)
        t = local_logs / ".w"
        t.touch()
        t.unlink()
        return local_logs
    except (PermissionError, OSError):
        p = get_app_data_dir() / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p


def log_scan_error(worker_name: str, file_path: str | Path, error: Exception | str) -> None:
    """
    Appends broken or corrupted media file details to logs/scan_errors.log.
    Safe against I/O exceptions and creates the logs directory if it does not exist.
    """
    try:
        log_dir = _get_log_dir()
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
    target_dir = Path(log_dir) if log_dir else _get_log_dir()
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


class UniversalToolTipFilter:
    """Placeholder to be initialized after PySide6 imports."""
    pass


def install_universal_tooltip_filter(app: Any = None) -> Any:
    """
    Installs a universal event filter intercepting all QTipLabel / ToolTip widgets across the app.
    Guarantees tooltips never render with system dark mode / black backgrounds regardless
    of local widget stylesheets or child hierarchy overrides.
    """
    from PySide6.QtCore import QEvent, QObject, Qt
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication, QWidget

    class _ToolTipFilter(QObject):
        def eventFilter(self, obj: QObject, event: QEvent) -> bool:
            if getattr(obj, "objectName", lambda: "")() == "qtooltip_label" or (
                isinstance(obj, QWidget) and obj.windowType() == Qt.WindowType.ToolTip
            ):
                if not getattr(obj, "_styled_universal", False):
                    obj._styled_universal = True
                    try:
                        pal = obj.palette()
                        pal.setColor(QPalette.ColorRole.Window, QColor("#FFFFFF"))
                        pal.setColor(QPalette.ColorRole.WindowText, QColor("#1E395B"))
                        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
                        pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#1E395B"))
                        obj.setPalette(pal)
                        obj.setStyleSheet(
                            "background-color: #FFFFFF !important; "
                            "color: #1E395B !important; "
                            "border: 1px solid #7DA2CE !important; "
                            "border-radius: 2px !important; "
                            "padding: 3px 6px !important; "
                            "font-family: 'Segoe UI', sans-serif !important; "
                            "font-size: 9pt !important;"
                        )
                    except Exception:
                        pass
            return False

    if app is None:
        app = QApplication.instance()
    if app is None:
        return None
    if not hasattr(app, "_universal_tooltip_filter"):
        filt = _ToolTipFilter(app)
        app.installEventFilter(filt)
        app._universal_tooltip_filter = filt
    return getattr(app, "_universal_tooltip_filter", None)

