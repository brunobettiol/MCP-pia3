import os
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    """Configurações da aplicação carregadas de variáveis de ambiente."""
    
    # Configurações do servidor
    APP_NAME: str = "MCP-API"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = Field(default=False)
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    
    # Diretório base do projeto
    BASE_DIR: str = str(Path(__file__).resolve().parent.parent.parent)
    
    # Configurações do Shopify
    SHOPIFY_STORE: str = Field(...)
    SHOPIFY_API_VERSION: str = Field(default="2024-10")
    SHOPIFY_STOREFRONT_TOKEN: str = Field(...)
    SHOPIFY_ADMIN_TOKEN: Optional[str] = None
    
    # Configurações de logging
    LOG_LEVEL: str = Field(default="INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 