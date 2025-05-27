from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

T = TypeVar('T')

class ErrorDetail(BaseModel):
    """Detalhes de um erro."""
    
    loc: List[str] = Field([], description="Localização do erro")
    msg: str = Field(..., description="Mensagem de erro")
    type: str = Field(..., description="Tipo de erro")

class ErrorResponse(BaseModel):
    """Modelo para resposta de erro."""
    
    success: bool = False
    message: str
    errors: Optional[List[ErrorDetail]] = None
    
class HealthCheck(BaseModel):
    """Modelo para verificação de saúde da API."""
    
    status: str
    version: str
    
class GenericResponse(GenericModel, Generic[T]):
    """Modelo genérico para respostas da API."""
    
    success: bool = True
    data: Optional[T] = None
    message: Optional[str] = None 