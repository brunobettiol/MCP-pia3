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

- `GET /api/v1/products` - Lista todos os produtos
- `GET /api/v1/products/{handle}` - Busca um produto pelo handle
- `GET /api/v1/products/ai/format` - Retorna produtos formatados para uso em IA
- `GET /api/v1/health` - Verificação de saúde da API

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
│   │       ├── health.py
│   │       └── product.py
│   ├── core/
│   │   ├── app.py
│   │   ├── config.py
│   │   └── logger.py
│   ├── models/
│   │   ├── common.py
│   │   └── product.py
│   └── services/
│       └── shopify_service.py
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