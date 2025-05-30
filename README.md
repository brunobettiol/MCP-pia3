# MCP API - Middleware Control Point para Shopify

API para atuar como um middleware entre chatbots e a plataforma Shopify, permitindo consultar produtos e formatar dados para uso em assistentes virtuais.

## Tecnologias Utilizadas

- Python 3.8+
- FastAPI
- Uvicorn
- Loguru
- Pydantic
- HTTPX

## Configuração

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/mcp-api.git
cd mcp-api
```

2. Crie um ambiente virtual e instale as dependências:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Copie o arquivo de exemplo de variáveis de ambiente:
```bash
cp env.example .env
```

4. Edite o arquivo `.env` com suas credenciais do Shopify:
```
SHOPIFY_STORE=sua-loja.myshopify.com
SHOPIFY_STOREFRONT_TOKEN=seu_token_storefront_aqui
```

5. Configure os arquivos de blog (opcional):
```bash
mkdir -p data/blogs
# Adicione seus arquivos JSON de blog na pasta data/blogs
```

## Executando Localmente

```bash
python server.py
```

Ou diretamente com o uvicorn:
```bash
uvicorn server:app --reload
```

A API estará disponível em `http://localhost:8000`
Documentação Swagger: `http://localhost:8000/docs`

## Endpoints Principais

### Produtos
- `GET /api/v1/products` - Lista todos os produtos
- `GET /api/v1/products/{handle}` - Busca um produto pelo handle
- `GET /api/v1/products/ai/format` - Retorna produtos formatados para uso em IA

### Blogs
- `GET /api/v1/blogs` - Lista todos os blogs
- `GET /api/v1/blogs/category/{category}` - Busca blogs por categoria
- `GET /api/v1/blogs/tag/{tag}` - Busca blogs por tag
- `GET /api/v1/blogs/search` - Busca blogs por texto
- `GET /api/v1/blogs/recommend` - Recomenda blogs com base em uma consulta
- `GET /api/v1/blogs/recommend/product/{product_handle}` - Recomenda blogs relacionados a um produto
- `GET /api/v1/blogs/ai/recommend` - Recomenda blogs formatados para uso em IA

### Outros
- `GET /api/v1/health` - Verificação de saúde da API

## Integração de Blogs

### Formato dos Arquivos

Os blogs devem ser armazenados como arquivos JSON na pasta `data/blogs`. Cada arquivo representa uma categoria de blogs e deve seguir o formato:

```json
[
  {
    "title": "Título do Blog",
    "content": "Conteúdo do blog...",
    "source_id": "/caminho/para/fonte",
    "created_at": "2025-04-29T00:00:00.000Z",
    "author": "Nome do Autor",
    "category": "categoria",
    "tags": ["tag1", "tag2", "tag3"]
  },
  ...
]
```

### Recomendação de Blogs para Chatbots

Para integrar recomendações de blog no seu chatbot:

1. Use o endpoint `/api/v1/blogs/ai/recommend?query=texto` para obter blogs recomendados com base no contexto da conversa.
2. Para recomendar blogs relacionados a produtos, use `/api/v1/blogs/recommend/product/{product_handle}`.

Exemplo de uso em um chatbot:
```python
async def get_blog_recommendations(query):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://sua-api.com/api/v1/blogs/ai/recommend?query={query}"
        )
        return response.json()
```

## Deploy no Railway

1. Faça login no Railway e crie um novo projeto
2. Conecte o repositório GitHub
3. Configure as variáveis de ambiente no Railway:
   - `SHOPIFY_STORE`
   - `SHOPIFY_STOREFRONT_TOKEN`
   - `SHOPIFY_API_VERSION` (opcional, padrão: 2024-10)
   - `DEBUG` (opcional, padrão: False)

O Railway detectará automaticamente o Procfile e fará o deploy da aplicação.

## Estrutura do Projeto

```
.
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── blog.py
│   │       ├── health.py
│   │       └── product.py
│   ├── core/
│   │   ├── app.py
│   │   ├── config.py
│   │   └── logger.py
│   ├── models/
│   │   ├── blog.py
│   │   ├── common.py
│   │   └── product.py
│   └── services/
│       ├── blog_service.py
│       └── shopify_service.py
├── data/
│   └── blogs/
│       └── *.json
├── logs/
├── .env
├── .env.example
├── Procfile
├── README.md
├── requirements.txt
└── server.py
```

## Licença

MIT 