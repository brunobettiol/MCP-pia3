from fastapi import APIRouter
from app.models.common import HealthCheck

router = APIRouter()

@router.get(
    "/",
    response_model=HealthCheck,
    summary="Verificação de saúde",
    description="Verifica se a API está funcionando corretamente."
)
async def health_check():
    """
    Endpoint para verificar a saúde da API.
    
    Returns:
        HealthCheck: Status da API.
    """
    return HealthCheck(status="ok", version="1.0.0") 