"""Конфиг логгера"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = "mlops_api", level: int = logging.INFO) -> logging.Logger:
    """
    Настройка логгера

    Args:
        name: Имя логгера
        level: Уровень логирования
        
    Returns:
        Настроенный экземпляр логгера
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

