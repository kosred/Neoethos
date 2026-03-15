import logging
import os
import queue
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from pathlib import Path


class ColorFormatter(logging.Formatter):
    """Highlight errors and warnings in the terminal."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    GRAY = "\033[90m"

    COLORS = {
        logging.DEBUG: GRAY,
        logging.INFO: BLUE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD + RED,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        log_fmt = f"{self.GRAY}%(asctime)s{self.RESET} [{color}%(levelname)s{self.RESET}] %(name)s: %(message)s"
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)


def setup_logging(verbose: bool = False) -> None:
    """HPC Optimized: Non-blocking structured logging with colored terminal output."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # 1. Base Stream Handler (Standard Out) with Colors
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColorFormatter())
    
    # 2. File Formatter (Standard Plain Text)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handlers = [stream_handler]
    
    # 3. Asynchronous File Handler
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / os.environ.get("LOG_FILE", "forex_bot.log")
    
    try:
        raw_file_handler = RotatingFileHandler(
            log_file, maxBytes=50*1024*1024, backupCount=3, encoding="utf-8"
        )
        raw_file_handler.setFormatter(file_formatter)
        
        # HPC FIX: Offload I/O to background thread
        log_queue = queue.Queue(-1) # Infinite buffer
        queue_handler = QueueHandler(log_queue)
        listener = QueueListener(log_queue, raw_file_handler)
        listener.start()
        
        handlers.append(queue_handler)
    except Exception as exc:
        logging.warning(f"Async logging setup failed: {exc}")

    logging.root.handlers = []
    for h in handlers:
        logging.root.addHandler(h)
    logging.root.setLevel(level)
    try:
        for name in (
            "httpcore",
            "httpx",
            "numba",
            "numba.core",
            "numba.core.byteflow",
            "openai",
            "urllib3",
            "matplotlib",
        ):
            lg = logging.getLogger(name)
            lg.setLevel(logging.WARNING)
            if name not in {"httpcore", "httpx"}:
                lg.propagate = False
    except Exception as e:
        logging.getLogger(__name__).warning(f"Log setup failed: {e}", exc_info=True)

    logging.captureWarnings(True)
