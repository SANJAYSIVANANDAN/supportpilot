import os


POSTGRES_USER = os.environ.get("POSTGRES_USER", "supportagent")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "changeme")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "supportagent")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")

DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "ai-support-agent")

KEYCLOAK_URL = os.environ.get("KEYCLOAK_URL", "http://keycloak:8080")
KEYCLOAK_REALM = os.environ.get("KEYCLOAK_REALM", "support")
KEYCLOAK_CLIENT_ID = os.environ.get("KEYCLOAK_CLIENT_ID", "support-api")
KEYCLOAK_CLIENT_SECRET = os.environ.get("KEYCLOAK_CLIENT_SECRET", "changeme")

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://mcp:8000")
API_URL = os.environ.get("API_URL", "http://api:8000")

REDIS_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
