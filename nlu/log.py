
import os
import logging

LEVEL = {
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'debug': logging.DEBUG
}

def bootstrap():

    level = os.environ.get('NLU_LOG_LEVEL')
    if isinstance(level, str) and level.lower() in LEVEL:
        level = LEVEL[level.lower()]
    else:
        level = logging.INFO

    # logging.basicConfig(level=level, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_formator = logging.Formatter('%(asctime)s [%(threadName)s] [%(levelname)s] [%(filename)s] [%(funcName)s] [%(lineno)s] [%(message)s]')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formator)

    logger = logging.getLogger('NLU')
    logger.addHandler(console_handler)
    logger.setLevel(level)

    return logger

LOG = bootstrap()
