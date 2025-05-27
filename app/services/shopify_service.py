import httpx
from typing import Dict, List, Optional, Any
from loguru import logger
from app.core.config import settings
from app.models.product import Product, ProductImage, ProductPrice, ProductVariant

class ShopifyService:
    """Serviço para comunicação com a API do Shopify."""
    
    def __init__(self):
        """Inicializa o serviço Shopify com as configurações da aplicação."""
        self.store = settings.SHOPIFY_STORE
        self.api_version = settings.SHOPIFY_API_VERSION
        self.storefront_token = settings.SHOPIFY_STOREFRONT_TOKEN
        self.admin_token = settings.SHOPIFY_ADMIN_TOKEN
        self.base_url = f"https://{self.store}/api/{self.api_version}"
        self.debug = settings.DEBUG
    
    async def _graphql_request(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executa uma requisição GraphQL para a API Storefront do Shopify.
        
        Args:
            query: A consulta GraphQL.
            variables: Variáveis para a consulta GraphQL.
            
        Returns:
            Resposta da API Shopify.
            
        Raises:
            Exception: Se ocorrer um erro na requisição.
        """
        if self.debug:
            logger.debug(f"Executando consulta GraphQL: {query}")
            logger.debug(f"Variáveis: {variables}")
        
        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Storefront-Access-Token": self.storefront_token
        }
        
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/graphql.json",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
                response.raise_for_status()
                result = response.json()
                
                if "errors" in result and result["errors"]:
                    error_messages = [error.get("message", "Unknown error") for error in result["errors"]]
                    logger.error(f"Erro na consulta GraphQL: {error_messages}")
                    raise Exception(f"Erro na API Shopify: {', '.join(error_messages)}")
                
                return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Erro HTTP na requisição Shopify: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Erro na API Shopify: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Erro na requisição Shopify: {str(e)}")
            raise Exception(f"Erro na conexão com a API Shopify: {str(e)}")
        except Exception as e:
            logger.error(f"Erro inesperado na comunicação com Shopify: {str(e)}")
            raise
    
    async def get_all_products(self) -> List[Product]:
        """
        Busca todos os produtos da loja Shopify.
        
        Returns:
            Lista de produtos.
        """
        products = []
        cursor = None
        has_next_page = True
        page_count = 0
        
        # Consulta GraphQL para buscar produtos
        query = """
        query getProducts($cursor: String) {
          products(first: 250, after: $cursor) {
            pageInfo {
              hasNextPage
            }
            edges {
              cursor
              node {
                id
                title
                handle
                description
                variants(first: 5) {
                  edges {
                    node {
                      id
                      title
                      availableForSale
                      sku
                      price {
                        amount
                        currencyCode
                      }
                    }
                  }
                }
                images(first: 5) {
                  edges {
                    node {
                      url
                      altText
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        try:
            logger.info("Iniciando busca de produtos no Shopify")
            
            while has_next_page:
                page_count += 1
                logger.info(f"Buscando página {page_count} de produtos")
                
                variables = {"cursor": cursor} if cursor else {}
                result = await self._graphql_request(query, variables)
                
                product_edges = result.get("data", {}).get("products", {}).get("edges", [])
                
                # Processar produtos da página atual
                for edge in product_edges:
                    node = edge.get("node", {})
                    
                    # Processar variantes
                    variants = []
                    for variant_edge in node.get("variants", {}).get("edges", []):
                        variant_node = variant_edge.get("node", {})
                        price_data = variant_node.get("price", {})
                        
                        price = ProductPrice(
                            amount=float(price_data.get("amount", 0)),
                            currency_code=price_data.get("currencyCode", "BRL")
                        )
                        
                        variant = ProductVariant(
                            id=variant_node.get("id", ""),
                            title=variant_node.get("title", ""),
                            price=price,
                            sku=variant_node.get("sku"),
                            available=variant_node.get("availableForSale", True)
                        )
                        
                        variants.append(variant)
                    
                    # Processar imagens
                    images = []
                    for image_edge in node.get("images", {}).get("edges", []):
                        image_node = image_edge.get("node", {})
                        image = ProductImage(
                            url=image_node.get("url", ""),
                            alt_text=image_node.get("altText")
                        )
                        images.append(image)
                    
                    # Criar objeto de produto
                    product = Product(
                        id=node.get("id", ""),
                        title=node.get("title", ""),
                        handle=node.get("handle", ""),
                        description=node.get("description", ""),
                        price=float(variants[0].price.amount) if variants else 0,
                        currency=variants[0].price.currency_code if variants else "BRL",
                        images=images,
                        variants=variants,
                        available=any(v.available for v in variants) if variants else False
                    )
                    
                    products.append(product)
                
                # Verificar se há mais páginas
                page_info = result.get("data", {}).get("products", {}).get("pageInfo", {})
                has_next_page = page_info.get("hasNextPage", False)
                
                if has_next_page and product_edges:
                    cursor = product_edges[-1].get("cursor")
            
            logger.info(f"Busca de produtos concluída. Total: {len(products)} produtos")
            return products
        
        except Exception as e:
            logger.error(f"Erro ao buscar produtos: {str(e)}")
            raise
    
    async def get_product_by_handle(self, handle: str) -> Optional[Product]:
        """
        Busca um produto específico pelo handle.
        
        Args:
            handle: O handle (slug) do produto.
            
        Returns:
            O produto encontrado ou None se não existir.
        """
        query = """
        query getProductByHandle($handle: String!) {
          product(handle: $handle) {
            id
            title
            handle
            description
            variants(first: 5) {
              edges {
                node {
                  id
                  title
                  availableForSale
                  sku
                  price {
                    amount
                    currencyCode
                  }
                }
              }
            }
            images(first: 5) {
              edges {
                node {
                  url
                  altText
                }
              }
            }
          }
        }
        """
        
        try:
            logger.info(f"Buscando produto com handle: {handle}")
            
            variables = {"handle": handle}
            result = await self._graphql_request(query, variables)
            
            product_data = result.get("data", {}).get("product")
            
            if not product_data:
                logger.warning(f"Produto não encontrado: {handle}")
                return None
            
            # Processar variantes
            variants = []
            for variant_edge in product_data.get("variants", {}).get("edges", []):
                variant_node = variant_edge.get("node", {})
                price_data = variant_node.get("price", {})
                
                price = ProductPrice(
                    amount=float(price_data.get("amount", 0)),
                    currency_code=price_data.get("currencyCode", "BRL")
                )
                
                variant = ProductVariant(
                    id=variant_node.get("id", ""),
                    title=variant_node.get("title", ""),
                    price=price,
                    sku=variant_node.get("sku"),
                    available=variant_node.get("availableForSale", True)
                )
                
                variants.append(variant)
            
            # Processar imagens
            images = []
            for image_edge in product_data.get("images", {}).get("edges", []):
                image_node = image_edge.get("node", {})
                image = ProductImage(
                    url=image_node.get("url", ""),
                    alt_text=image_node.get("altText")
                )
                images.append(image)
            
            # Criar objeto de produto
            product = Product(
                id=product_data.get("id", ""),
                title=product_data.get("title", ""),
                handle=product_data.get("handle", ""),
                description=product_data.get("description", ""),
                price=float(variants[0].price.amount) if variants else 0,
                currency=variants[0].price.currency_code if variants else "BRL",
                images=images,
                variants=variants,
                available=any(v.available for v in variants) if variants else False
            )
            
            logger.info(f"Produto encontrado: {product.title}")
            return product
        
        except Exception as e:
            logger.error(f"Erro ao buscar produto por handle: {str(e)}")
            raise

    def format_products_for_ai(self, products: List[Product]) -> List[Dict[str, str]]:
        """
        Formata produtos para uso em um chatbot de IA.
        
        Args:
            products: Lista de produtos a serem formatados.
            
        Returns:
            Lista de produtos formatados para IA.
        """
        logger.info(f"Formatando {len(products)} produtos para IA")
        
        formatted_products = []
        
        for product in products:
            # Limpar descrição HTML
            clean_description = product.description or ""
            clean_description = clean_description.replace("<p>", "").replace("</p>", " ")
            clean_description = clean_description.replace("<br>", " ").replace("<br/>", " ")
            
            # Remover todas as tags HTML restantes
            import re
            clean_description = re.sub(r'<[^>]+>', '', clean_description)
            
            # Limitar tamanho da descrição
            if clean_description and len(clean_description) > 200:
                clean_description = clean_description[:197] + "..."
            
            formatted_product = {
                "name": product.title,
                "price": f"{product.currency} {product.price}",
                "description": clean_description,
                "url": f"https://{self.store}/products/{product.handle}",
                "image": product.images[0].url if product.images else "",
                "available": "Disponível" if product.available else "Indisponível"
            }
            
            formatted_products.append(formatted_product)
        
        return formatted_products 