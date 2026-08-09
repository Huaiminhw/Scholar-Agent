import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: Path | None = None, level: str = 'INFO') -> None:
    logger.remove()
    logger.add(sys.stderr, level=level,
               format='<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>')
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / 'agent_{time:YYYY-MM-DD}.log',
                   level='DEBUG', rotation='10 MB', retention='7 days',
                   format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}')
