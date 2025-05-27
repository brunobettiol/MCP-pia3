from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.core.config import settings
from app.api.routes import router as api_router
from app.core.logger import setup_logging

def create_app() -> FastAPI:
    """Cria e configura a aplicação FastAPI."""
    
    # Configurar o logger
    setup_logging()
    
    # Criar a aplicação FastAPI
    app = FastAPI(
        title=settings.APP_NAME,
        description="API do Middleware Control Point (MCP) para integração com Shopify",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Em produção, especificar origens permitidas
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Incluir rotas da API
    app.include_router(api_router, prefix=settings.API_PREFIX)
    
    # Personalizar esquema OpenAPI
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=f"{settings.APP_NAME} API",
            version="1.0.0",
            description="API do Middleware Control Point (MCP) para integração com Shopify",
            routes=app.routes,
        )
        
        # Personalizar esquema OpenAPI
        openapi_schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": f"Bem-vindo à {settings.APP_NAME} API",
            "docs": "/docs",
            "version": "1.0.0"
        }
    
    return app 