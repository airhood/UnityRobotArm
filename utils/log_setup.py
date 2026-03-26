"""
Shared logging setup — console + timestamped file in logs/.
"""

import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")


def setup(name: str) -> logging.Logger:
    """
    Configure root logger with:
      - StreamHandler  (console)
      - FileHandler    (logs/<name>_YYYY-MM-DD_HH-MM-SS.log)

    Returns a named logger.
    """
    LOG_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file  = LOG_DIR / f"{name}_{timestamp}.log"

    fmt = "[%(asctime)s] %(levelname)s %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    logger = logging.getLogger(name)
    logger.info(f"Log file: {log_file}")
    return logger
