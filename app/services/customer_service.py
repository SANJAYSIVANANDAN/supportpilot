from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Customer, Ticket, TicketStatus, TicketPriority


def _get_session() -> Session:
    return SessionLocal()


def get_customer_by_email(email: str) -> dict | None:
    session = _get_session()
    try:
        customer = session.query(Customer).filter(Customer.email == email).first()
        if not customer:
            return None
        return {
            "id": customer.id,
            "name": customer.name,
            "email": customer.email,
            "phone": customer.phone,
            "company": customer.company,
            "created_at": customer.created_at.isoformat(),
        }
    finally:
        session.close()


def get_customer_by_id(customer_id: int) -> dict | None:
    session = _get_session()
    try:
        customer = session.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            return None
        return {
            "id": customer.id,
            "name": customer.name,
            "email": customer.email,
            "phone": customer.phone,
            "company": customer.company,
            "created_at": customer.created_at.isoformat(),
        }
    finally:
        session.close()


def list_tickets_for_customer(customer_id: int) -> list[dict]:
    session = _get_session()
    try:
        tickets = (
            session.query(Ticket)
            .filter(Ticket.customer_id == customer_id)
            .order_by(Ticket.created_at.desc())
            .all()
        )
        return [
            {
                "id": t.id,
                "subject": t.subject,
                "description": t.description,
                "status": t.status.value,
                "priority": t.priority.value,
                "assigned_to": t.assigned_to,
                "resolution": t.resolution,
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat(),
            }
            for t in tickets
        ]
    finally:
        session.close()


def get_ticket_by_id(ticket_id: int) -> dict | None:
    session = _get_session()
    try:
        ticket = session.query(Ticket).filter(Ticket.id == ticket_id).first()
        if not ticket:
            return None
        return {
            "id": ticket.id,
            "customer_id": ticket.customer_id,
            "subject": ticket.subject,
            "description": ticket.description,
            "status": ticket.status.value,
            "priority": ticket.priority.value,
            "assigned_to": ticket.assigned_to,
            "resolution": ticket.resolution,
            "created_at": ticket.created_at.isoformat(),
            "updated_at": ticket.updated_at.isoformat(),
        }
    finally:
        session.close()


def search_tickets(
    status: str | None = None,
    priority: str | None = None,
    customer_id: int | None = None,
) -> list[dict]:
    session = _get_session()
    try:
        query = session.query(Ticket)
        if status:
            query = query.filter(Ticket.status == TicketStatus(status))
        if priority:
            query = query.filter(Ticket.priority == TicketPriority(priority))
        if customer_id:
            query = query.filter(Ticket.customer_id == customer_id)
        tickets = query.order_by(Ticket.created_at.desc()).limit(50).all()
        return [
            {
                "id": t.id,
                "customer_id": t.customer_id,
                "subject": t.subject,
                "status": t.status.value,
                "priority": t.priority.value,
                "assigned_to": t.assigned_to,
                "created_at": t.created_at.isoformat(),
            }
            for t in tickets
        ]
    finally:
        session.close()


def create_ticket(
    customer_id: int,
    subject: str,
    description: str,
    priority: str = "medium",
) -> dict:
    session = _get_session()
    try:
        ticket = Ticket(
            customer_id=customer_id,
            subject=subject,
            description=description,
            priority=TicketPriority(priority),
        )
        session.add(ticket)
        session.commit()
        session.refresh(ticket)
        return {
            "id": ticket.id,
            "customer_id": ticket.customer_id,
            "subject": ticket.subject,
            "status": ticket.status.value,
            "priority": ticket.priority.value,
            "created_at": ticket.created_at.isoformat(),
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def update_ticket_status(ticket_id: int, status: str, resolution: str | None = None) -> dict | None:
    session = _get_session()
    try:
        ticket = session.query(Ticket).filter(Ticket.id == ticket_id).first()
        if not ticket:
            return None
        ticket.status = TicketStatus(status)
        if resolution:
            ticket.resolution = resolution
        session.commit()
        session.refresh(ticket)
        return {
            "id": ticket.id,
            "status": ticket.status.value,
            "resolution": ticket.resolution,
            "updated_at": ticket.updated_at.isoformat(),
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def escalate_ticket(ticket_id: int, assigned_to: str, reason: str) -> dict | None:
    session = _get_session()
    try:
        ticket = session.query(Ticket).filter(Ticket.id == ticket_id).first()
        if not ticket:
            return None
        ticket.status = TicketStatus.escalated
        ticket.assigned_to = assigned_to
        ticket.description = ticket.description + f"\n\n--- Escalation ---\n{reason}"
        session.commit()
        session.refresh(ticket)
        return {
            "id": ticket.id,
            "status": ticket.status.value,
            "assigned_to": ticket.assigned_to,
            "updated_at": ticket.updated_at.isoformat(),
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
