from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl

class ProductImage(BaseModel):
    """Modelo para representar uma imagem de produto."""
    
    url: HttpUrl
    alt_text: Optional[str] = None

class ProductPrice(BaseModel):
    """Modelo para representar o preço de um produto."""
    
    amount: float
    currency_code: str = "BRL"

class ProductVariant(BaseModel):
    """Modelo para representar uma variante de produto."""
    
    id: str
    title: str
    price: ProductPrice
    sku: Optional[str] = None
    available: bool = True
    
class Product(BaseModel):
    """Modelo para representar um produto."""
    
    id: str
    title: str
    handle: str
    description: Optional[str] = None
    price: float
    currency: str = "USD"
    images: List[ProductImage] = []
    variants: List[ProductVariant] = []
    available: bool = True
    tags: List[str] = []
    
class ProductList(BaseModel):
    """Modelo para representar uma lista de produtos."""
    
    products: List[Product]
    total: int = Field(..., description="Número total de produtos")
    
class ProductResponse(BaseModel):
    """Modelo para resposta da API de produtos."""
    
    success: bool = True
    data: ProductList
    message: Optional[str] = None 