# OntoDesk

A customer-support agent whose knowledge is a Neo4j ontology: it only
grows the graph when it sees a genuinely new, LLM-verified kind of
problem. See [`kg_agent/README.md`](kg_agent/README.md) for a terser setup/run
reference.

All the code lives in [`kg_agent/`](kg_agent/); this root `README.md`,
`requirements.txt`, and `.env.example` just mirror its dependencies/config
for convenience at the repo root.

## Quick start (Windows PowerShell)

```powershell
# 1. Create venv & install deps
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. Configure
copy kg_agent\.env.example kg_agent\.env
# edit kg_agent\.env with your OPENAI_API_KEY (NEO4J_* already default to
# a local instance at bolt://localhost:7687)

# 3. Start a local Neo4j (Community Edition, requires JDK 17+)
#    https://neo4j.com/deployment-center/
.\bin\neo4j-admin.bat dbms set-initial-password ontodesk123   # first time only
.\bin\neo4j.bat console

# 4. Seed the knowledge graph with a few canonical issues
.venv\Scripts\python -m kg_agent.seed

# 5. Try it out
.venv\Scripts\python -m kg_agent.test_manual            # no-UI smoke test
.venv\Scripts\streamlit run kg_agent\dashboard.py        # interactive dashboard
.venv\Scripts\python -m kg_agent.mcp_server              # as a private MCP server

# 6. Run tests
.venv\Scripts\python -m pytest kg_agent\tests -q
```

## MCP server

`kg_agent/mcp_server.py` exposes OntoDesk over stdio (no network port) —
only a process that spawns it directly can reach it. Register it in an
MCP client's config (e.g. Claude Desktop):

```json
{
  "mcpServers": {
    "ontodesk": {
      "command": "C:\\Users\\sanja\\projects\\supportpilot\\.venv\\Scripts\\python.exe",
      "args": ["-m", "kg_agent.mcp_server"],
      "cwd": "C:\\Users\\sanja\\projects\\supportpilot"
    }
  }
}
```

Tools: `ask_support(query)`, `graph_stats()`. See
[`kg_agent/GUIDE.md`](kg_agent/GUIDE.md) §3.4 for what each returns.
