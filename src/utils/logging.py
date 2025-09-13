import logging
import os


def get_logger(name: str, level: str | int | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    lvl = level or os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(lvl)
    return logger

