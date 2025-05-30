from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class BlogEntry(BaseModel):
    """Modelo para representar uma entrada de blog."""
    
    title: str
    content: str
    source_id: str
    created_at: datetime
    author: str
    category: str
    tags: List[str] = []

class BlogList(BaseModel):
    """Modelo para representar uma lista de blogs."""
    
    blogs: List[BlogEntry]
    total: int = Field(..., description="Número total de blogs")
    
class BlogResponse(BaseModel):
    """Modelo para resposta da API de blogs."""
    
    success: bool = True
    data: BlogList
    message: Optional[str] = None

class BlogRecommendation(BaseModel):
    """Modelo para recomendação de blog."""
    
    query: str
    recommended_blogs: List[BlogEntry]
    relevance_scores: Optional[List[float]] = None 