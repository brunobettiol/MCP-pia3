import sys
import logging
from loguru import logger
from app.core.config import settings

class InterceptHandler(logging.Handler):
    """
    Interceptor de logs padrão do Python para usar o Loguru.
    """
    
    def emit(self, record):
        # Obter o nível correspondente do Loguru
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Encontrar o frame de origem do log
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging():
    """Configura o sistema de logging usando Loguru."""
    
    # Remover todos os manipuladores padrão
    logger.remove()
    
    # Adicionar manipulador para stdout com formatação personalizada
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True,
    )
    
    # Adicionar manipulador para arquivo de log
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
    )
    
    # Interceptar logs padrão do Python
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    # Substituir manipuladores para bibliotecas comuns
    for logger_name in ("uvicorn", "uvicorn.error", "fastapi"):
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
    
    logger.info("Sistema de logging configurado com sucesso")
    
    return logger 