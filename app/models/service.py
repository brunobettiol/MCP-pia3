from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class ServiceContact(BaseModel):
    phone: str
    email: str
    website: str
    internalLink: str


class ServiceAddress(BaseModel):
    street: str
    city: str
    state: str
    zipCode: str


class ServiceProvider(BaseModel):
    id: str
    name: str
    type: str
    types: List[str] = []
    description: str
    detailedDescription: str
    contact: ServiceContact
    address: ServiceAddress
    serviceAreas: List[str] = []
    serviceRegions: List[str] = []
    services: List[str] = []
    specialties: List[str] = []
    credentials: List[str] = []
    languages: List[str] = []
    paymentOptions: List[str] = []
    yearsInBusiness: str = ""
    featured: bool = False
    lastUpdated: str


class ServiceList(BaseModel):
    providers: List[ServiceProvider]
    total: int


class ServiceResponse(BaseModel):
    success: bool
    data: Optional[ServiceList] = None
    message: Optional[str] = None 