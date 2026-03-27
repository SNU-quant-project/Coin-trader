"""
로그 모듈
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

_loggers = {}


def get_logger(name, level=None, log_dir=None):
    if name in _loggers:
        return _loggers[name]

    try:
        from config.config_loader import Config
        config = Config()
        if level is None:
            level = config.log_level
        if log_dir is None:
            log_dir = config.log_dir
    except Exception:
        if level is None:
            level = "INFO"
        if log_dir is None:
            log_dir = "logs"

    logger = logging.getLogger(f"crypto.{name}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        _loggers[name] = logger
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(module_name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_ModuleNameFilter(name))
    logger.addHandler(console_handler)

    try:
        os.makedirs(log_dir, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"{today}.log")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_ModuleNameFilter(name))
        logger.addHandler(file_handler)
    except Exception:
        pass

    _loggers[name] = logger
    return logger


class _ModuleNameFilter(logging.Filter):
    def __init__(self, module_name):
        super().__init__()
        self.module_name = module_name

    def filter(self, record):
        record.module_name = self.module_name
        return True
