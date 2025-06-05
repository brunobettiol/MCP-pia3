from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class BlogEntry(BaseModel):
    """Modelo para representar uma entrada de blog."""
    
    id: str
    title: str
    content: str
    author: str
    category: str
    subcategory: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = []
    featured: bool = False
    contentType: Optional[str] = None
    createdAt: datetime
    updatedAt: Optional[datetime] = None
    publishedDate: Optional[datetime] = None
    
    # For backward compatibility
    @property
    def source_id(self) -> str:
        return self.id
    
    @property
    def created_at(self) -> datetime:
        return self.createdAt

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