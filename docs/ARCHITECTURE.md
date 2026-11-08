# AI Support Agent — Architecture

## Services

| Service   | Container | Port (host) | Port (container) |
|-----------|-----------|-------------|-------------------|
| Frontend  | frontend  | 8501        | 8501              |
| API       | api       | 8000        | 8000              |
| MCP Server| mcp       | 8001        | 8000              |
| PostgreSQL| postgres  | 5433        | 5432              |
| Redis     | redis     | 6379        | 6379              |
| Keycloak  | keycloak  | 8080        | 8080              |

## Data Flow

1. User authenticates via Keycloak (username/password → JWT)
2. Streamlit frontend sends chat message + JWT to FastAPI `/chat`
3. FastAPI validates JWT, checks RBAC role, invokes LangGraph agent
4. Agent uses MCP tools (via HTTP) for reads, local LangChain tools for writes
5. All reads go through `app/services/customer_service.py` — no duplicated SQL
6. Conversation memory persisted in Redis (7-day TTL, refresh on read)
7. LangSmith receives traces for observability

## Auth

- Keycloak 26.2, RS256 JWTs
- Realm: `support`
- Roles: `agent`, `admin`
- Users: `agent1` / `agent1pass`, `admin1` / `admin1pass`

## Quick Start

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env
docker compose up --build
```

Open http://localhost:8501, log in as `agent1` / `agent1pass`.
