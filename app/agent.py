import json
import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    REDIS_URL,
    REDIS_TTL_SECONDS,
    MCP_SERVER_URL,
)
from app.services.customer_service import (
    create_ticket,
    update_ticket_status,
    escalate_ticket,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful customer-support AI agent. You assist customers and support agents by looking up customer information, managing support tickets, and escalating issues when needed.

RULES:
- NEVER fabricate customer data, ticket details, or any business information.
- ALWAYS use the provided tools to look up information before answering.
- If a tool returns an error or no data, tell the user honestly.
- For read operations (looking up customers, tickets), use the MCP tools.
- For write operations (creating tickets, updating status, escalating), use the local tools.
- Be concise, professional, and empathetic.
"""


# --- Write tools (local LangChain tools) ---

@tool
def create_support_ticket(
    customer_id: int, subject: str, description: str, priority: str = "medium"
) -> str:
    """Create a new support ticket for a customer. Priority: low, medium, high, urgent."""
    try:
        result = create_ticket(customer_id, subject, description, priority)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def update_ticket(ticket_id: int, status: str, resolution: str | None = None) -> str:
    """Update the status of a ticket. Valid statuses: open, in_progress, escalated, resolved, closed."""
    try:
        result = update_ticket_status(ticket_id, status, resolution)
        if result is None:
            return json.dumps({"error": "Ticket not found"})
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def escalate_support_ticket(ticket_id: int, assigned_to: str, reason: str) -> str:
    """Escalate a ticket to a human agent with a reason."""
    try:
        result = escalate_ticket(ticket_id, assigned_to, reason)
        if result is None:
            return json.dumps({"error": "Ticket not found"})
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- MCP read tools (call the MCP server over HTTP) ---

async def _call_mcp_tool(tool_name: str, arguments: dict) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{MCP_SERVER_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
                "id": 1,
            },
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            for line in resp.text.strip().split("\n"):
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "result" in data:
                        content = data["result"].get("content", [])
                        if content:
                            return content[0].get("text", json.dumps(data["result"]))
            return json.dumps({"error": "No result in SSE stream"})

        data = resp.json()
        if "result" in data:
            content = data["result"].get("content", [])
            if content:
                return content[0].get("text", json.dumps(data["result"]))
        return json.dumps(data)


@tool
async def lookup_customer_by_email(email: str) -> str:
    """Look up a customer by their email address."""
    return await _call_mcp_tool("lookup_customer_by_email", {"email": email})


@tool
async def lookup_customer_by_id(customer_id: int) -> str:
    """Look up a customer by their numeric ID."""
    return await _call_mcp_tool("lookup_customer_by_id", {"customer_id": customer_id})


@tool
async def get_ticket(ticket_id: int) -> str:
    """Get full details of a support ticket by ID."""
    return await _call_mcp_tool("get_ticket", {"ticket_id": ticket_id})


@tool
async def list_customer_tickets(customer_id: int) -> str:
    """List all support tickets for a customer."""
    return await _call_mcp_tool("list_customer_tickets", {"customer_id": customer_id})


@tool
async def search_support_tickets(
    status: str | None = None,
    priority: str | None = None,
    customer_id: int | None = None,
) -> str:
    """Search tickets with optional filters."""
    args: dict[str, Any] = {}
    if status:
        args["status"] = status
    if priority:
        args["priority"] = priority
    if customer_id:
        args["customer_id"] = customer_id
    return await _call_mcp_tool("search_support_tickets", args)


# --- Graph construction ---

ALL_TOOLS = [
    lookup_customer_by_email,
    lookup_customer_by_id,
    get_ticket,
    list_customer_tickets,
    search_support_tickets,
    create_support_ticket,
    update_ticket,
    escalate_support_ticket,
]


def _build_graph():
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: MessagesState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        return {"messages": [llm_with_tools.invoke(messages)]}

    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph


_graph = _build_graph()


async def get_checkpointer():
    saver = AsyncRedisSaver.from_conn_string(REDIS_URL)
    await saver.asetup()
    return saver


async def run_agent(user_message: str, thread_id: str) -> str:
    start = time.monotonic()
    checkpointer = await get_checkpointer()
    graph = _graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )

    elapsed = time.monotonic() - start
    logger.info("Agent run completed in %.2fs for thread %s", elapsed, thread_id)

    ai_messages = [m for m in result["messages"] if m.type == "ai" and m.content]
    if ai_messages:
        return ai_messages[-1].content
    return "I'm sorry, I wasn't able to generate a response."
