import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

from app.services.customer_service import (
    get_customer_by_email,
    get_customer_by_id,
    get_ticket_by_id,
    list_tickets_for_customer,
    search_tickets,
)

mcp = FastMCP("support-tools", host="0.0.0.0", port=8000)


@mcp.tool()
def lookup_customer_by_email(email: str) -> str:
    """Look up a customer by their email address. Returns customer details or not-found message."""
    result = get_customer_by_email(email)
    if result is None:
        return json.dumps({"error": "Customer not found"})
    return json.dumps(result)


@mcp.tool()
def lookup_customer_by_id(customer_id: int) -> str:
    """Look up a customer by their numeric ID. Returns customer details or not-found message."""
    result = get_customer_by_id(customer_id)
    if result is None:
        return json.dumps({"error": "Customer not found"})
    return json.dumps(result)


@mcp.tool()
def get_ticket(ticket_id: int) -> str:
    """Get full details of a single support ticket by its ID."""
    result = get_ticket_by_id(ticket_id)
    if result is None:
        return json.dumps({"error": "Ticket not found"})
    return json.dumps(result)


@mcp.tool()
def list_customer_tickets(customer_id: int) -> str:
    """List all support tickets for a given customer, newest first."""
    results = list_tickets_for_customer(customer_id)
    return json.dumps(results)


@mcp.tool()
def search_support_tickets(
    status: str | None = None,
    priority: str | None = None,
    customer_id: int | None = None,
) -> str:
    """Search tickets with optional filters. Valid statuses: open, in_progress, escalated, resolved, closed. Valid priorities: low, medium, high, urgent."""
    results = search_tickets(status=status, priority=priority, customer_id=customer_id)
    return json.dumps(results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
