import json
import logging

from langchain_core.tools import tool

from app.services.customer_service import escalate_ticket

logger = logging.getLogger(__name__)


@tool
def escalate_to_human(ticket_id: int, assigned_to: str, reason: str) -> str:
    """Escalate a support ticket to a human agent. Use this when the issue is too complex for AI resolution, the customer explicitly requests a human, or the issue involves billing/refunds that require manual approval.

    Args:
        ticket_id: The ticket to escalate.
        assigned_to: The human agent or team to assign to (e.g. 'billing-team', 'senior-support').
        reason: Why the ticket is being escalated.
    """
    try:
        result = escalate_ticket(ticket_id, assigned_to, reason)
        if result is None:
            return json.dumps({"error": "Ticket not found"})
        logger.info("Ticket %d escalated to %s: %s", ticket_id, assigned_to, reason)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
