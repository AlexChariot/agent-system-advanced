"""
Centralized logging configuration for the agent system.

Call `setup_logging()` once at startup (in main.py) to apply
blue-colored INFO logs across all agents.

Color codes are ANSI escape sequences, supported by all modern terminals.
No external dependency required.
"""

import logging

# ANSI color codes
BLUE   = "\033[34m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

LEVEL_COLORS = {
    logging.DEBUG:    "",        # no color for debug
    logging.INFO:     BLUE,
    logging.WARNING:  YELLOW,
    logging.ERROR:    RED,
    logging.CRITICAL: BOLD + RED,
}


class ColoredFormatter(logging.Formatter):
    """
    Log formatter that applies ANSI color codes based on the log level.

    Format: [LEVEL] message
    INFO and below use blue, WARNING uses yellow, ERROR/CRITICAL use red.
    """

    BASE_FORMAT = "[%(levelname)s] %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        color = LEVEL_COLORS.get(record.levelno, "")
        formatter = logging.Formatter(f"{color}{self.BASE_FORMAT}{RESET}")
        return formatter.format(record)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with the colored formatter.

    Should be called once at application startup, before any agent runs.
    All loggers in the agent_system package will inherit this configuration.

    Args:
        level (int): Logging level (default: logging.INFO).
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())

    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)