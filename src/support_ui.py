"""Customer handoff and the staff queue inside the existing Streamlit app."""

import json

import streamlit as st

from src.config import Settings
from src.security import Principal, authenticate, rate_limit
from src.store import Store
from src.tickets import STATUSES, TicketService


def services():
    settings = Settings.from_env()
    store = Store(settings.state_dir / "control.sqlite3")
    return settings, store, TicketService(store, settings)


def remember_ticket(ticket):
    references = st.session_state.setdefault("support_tickets", [])
    if ticket["reference"] not in references:
        references.append(ticket["reference"])


def render_customer_support():
    with st.expander("Talk to a human"):
        with st.form("request-human-support", clear_on_submit=True):
            question = st.text_input("Describe your issue", max_chars=2000)
            if st.form_submit_button("Create support ticket"):
                from src.rag_memory import request_human_support
                try:
                    ticket = request_human_support(question, st.session_state.chat_history, st.session_state.support_session)
                    remember_ticket(ticket)
                    st.rerun()
                except (ValueError, PermissionError):
                    st.warning("Enter your issue or wait a moment before trying again.")
                except Exception:
                    st.error("The ticket could not be created. Please try again shortly.")
    references = st.session_state.get("support_tickets", [])
    if not references:
        return
    settings, _, tickets = services()
    st.subheader("Your support tickets")
    for reference in reversed(references[-10:]):
        try:
            view = tickets.customer_view(reference, st.session_state.support_session)
        except PermissionError:
            continue
        with st.expander(f"{reference} · {view['status'].replace('_', ' ').title()}"):
            for message in view["messages"]:
                st.caption("Support team" if message["author"] == "SUPPORT" else "You")
                st.write(message["body"])
            if not view["messages"]:
                st.caption("Waiting for a support team member to respond.")
            st.button("Refresh ticket", key="refresh-" + reference)
            if settings.ticket_encryption_key and view["status"] not in {"RESOLVED", "CLOSED"}:
                with st.form("contact-" + reference, clear_on_submit=True):
                    contact = st.text_input("Email or phone for follow-up (optional)", max_chars=254)
                    if st.form_submit_button("Save contact"):
                        try:
                            tickets.contact(reference, contact, st.session_state.support_session)
                            st.success("Contact saved securely for the support team.")
                        except (ValueError, PermissionError):
                            st.warning("Check the contact details and try again.")
            if view["status"] not in {"RESOLVED", "CLOSED"}:
                with st.form("customer-reply-" + reference, clear_on_submit=True):
                    reply = st.text_input("Add details for the support team", max_chars=4000)
                    if st.form_submit_button("Send to support"):
                        try:
                            tickets.reply(reference, reply, session=st.session_state.support_session)
                            st.rerun()
                        except (ValueError, PermissionError):
                            st.warning("Message could not be saved. Check the ticket and try again.")


def render_staff_workspace():
    settings, store, tickets = services()
    principal = st.session_state.get("staff_principal")
    if not isinstance(principal, Principal):
        st.subheader("Staff sign in")
        with st.form("staff-login", clear_on_submit=True):
            role = st.selectbox("Role", ["support", "reviewer", "admin"])
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in to staff workspace"):
                try:
                    rate_limit(store, "staff-login:" + st.session_state.support_session, 5)
                    rate_limit(store, "staff-login-role:" + role, 20)
                    st.session_state.staff_principal = authenticate(role, password)
                    store.audit(role, "staff.login", role)
                    st.rerun()
                except PermissionError:
                    st.error("Sign-in failed or too many attempts. Check your staff credentials.")
        return
    st.caption(f"Signed in as {principal.name}")
    if st.button("Sign out of staff workspace"):
        st.session_state.pop("staff_principal", None)
        st.rerun()
    if principal.role in {"support", "admin"}:
        render_staff_tickets(tickets, principal)
    if principal.role in {"reviewer", "admin"}:
        render_staff_faqs(settings, store, principal)


def render_staff_tickets(tickets, principal):
    st.subheader("Human support queue")
    include_closed = st.checkbox("Include resolved and closed tickets")
    rows = tickets.list(principal, include_closed)
    if not rows:
        st.info("No support tickets in this queue.")
        return
    labels = {r["reference"]: f"{r['reference']} · {r['status']} · {json.loads(r['payload'])['question'][:80]}" for r in rows}
    reference = st.selectbox("Support ticket", list(labels), format_func=labels.get)
    ticket = tickets.detail(reference, principal)
    st.write(ticket["payload"]["question"])
    if ticket["payload"]["last_answer"]:
        st.caption("Last automated answer")
        st.write(ticket["payload"]["last_answer"])
    if ticket["contact"]:
        st.write("Contact:", ticket["contact"])
    with st.expander("Conversation context"):
        for message in ticket["payload"]["history"]:
            st.caption(message["role"].capitalize())
            st.write(message["content"])
    for message in ticket["messages"]:
        st.caption("Support" if message["author"] == "SUPPORT" else "Customer")
        st.write(message["body"])
    with st.form("manage-" + reference, clear_on_submit=True):
        statuses = sorted(STATUSES)
        status = st.selectbox("Status", statuses, index=statuses.index(ticket["status"]))
        assignee = st.text_input("Assigned team member", value=ticket["assignee"] or "", max_chars=100)
        reply = st.text_area("Reply to customer", max_chars=4000)
        if st.form_submit_button("Save ticket and send reply"):
            try:
                if reply.strip():
                    tickets.reply(reference, reply, principal=principal)
                tickets.update(reference, status, assignee, principal)
                st.rerun()
            except (ValueError, PermissionError) as error:
                st.warning(str(error))


def render_staff_faqs(settings, store, principal):
    from src.faq import FAQService
    from src.jobs import JobQueue
    from src.ollama_client import OllamaClient
    from src.verification import GroundingVerifier
    st.subheader("FAQ generation and review")
    st.button("Refresh FAQ drafts", key="refresh-faq-drafts")
    client = OllamaClient(settings)
    service = FAQService(store, settings, client, GroundingVerifier(client, settings.verifier_model))
    if principal.role == "admin":
        chunks = store.rows("""SELECT c.id,c.metadata FROM chunks c JOIN document_versions v ON v.id=c.version_id
            JOIN documents d ON d.id=c.document_id WHERE d.deleted=0 AND v.status IN ('INDEXED','EXTRACTED')
            AND v.version=(SELECT MAX(v2.version) FROM document_versions v2 WHERE v2.document_id=d.id
                AND v2.status IN ('INDEXED','EXTRACTED')) ORDER BY d.source_key,c.id LIMIT 500""")
        if chunks:
            labels = {c["id"]: f"{json.loads(c['metadata'])['source']} · page {json.loads(c['metadata']).get('page_number') or 'n/a'} · {c['id'][-8:]}" for c in chunks}
            with st.form("generate-faqs"):
                chunk_id = st.selectbox("Source section", list(labels), format_func=labels.get)
                count = st.number_input("Questions to generate", min_value=1, max_value=10, value=settings.faq_per_chunk)
                if st.form_submit_button("Generate draft FAQs"):
                    job = JobQueue(store).enqueue("generate_faq", {"chunk_id": chunk_id, "count": count}, principal)
                    st.success(f"Generation queued: {job}")
        jobs = store.rows("SELECT id,status,error FROM jobs WHERE kind='generate_faq' ORDER BY created_at DESC LIMIT 5")
        for job in jobs:
            st.caption(f"Generation {job['id'][-8:]}: {job['status'].lower()}" +
                       (f" ({job['error']})" if job["error"] else ""))
        runs = store.rows("SELECT detail FROM generation_runs ORDER BY created_at DESC LIMIT 1")
        if runs:
            detail = json.loads(runs[0]["detail"])
            st.caption(f"Latest generation: {len(detail['accepted'])} drafts, {len(detail['rejected'])} rejected candidates.")
    rows = store.rows("SELECT f.*,d.source_key FROM faqs f JOIN documents d ON d.id=f.document_id WHERE d.deleted=0 ORDER BY f.updated_at DESC LIMIT 200")
    if not rows:
        st.info("No FAQ drafts yet. Generated drafts will appear here for review.")
        return
    by_id = {r["id"]: r for r in rows}
    faq_id = st.selectbox("FAQ to review", list(by_id), format_func=lambda identity:
        f"{by_id[identity]['status']} · {json.loads(by_id[identity]['body'])['question']}")
    row, body = by_id[faq_id], json.loads(by_id[faq_id]["body"])
    st.caption(f"{row['source_key']} · page {body.get('source_page') or 'n/a'} · {row['status']}")
    st.write(body["question"])
    st.write(body["answer"])
    st.caption("Exact supporting source evidence")
    st.write(body["supporting_text"])
    with st.expander("Full source section"):
        source = store.rows("SELECT text FROM chunks WHERE id=?", (row["chunk_id"],))
        if source:
            st.write(source[0]["text"])
    if row["status"] != "OUTDATED":
        with st.form("edit-" + faq_id):
            question = st.text_input("Question", value=body["question"], max_chars=500)
            answer = st.text_area("Answer", value=body["answer"], max_chars=3000)
            evidence = st.text_area("Supporting source quote", value=body["supporting_text"], max_chars=6000)
            if st.form_submit_button("Save edits for review"):
                try:
                    service.edit(faq_id, {"question": question, "answer": answer, "supporting_text": evidence}, principal)
                    st.success("Edits validated. FAQ returned to review.")
                except (ValueError, PermissionError, RuntimeError) as error:
                    st.warning(str(error))
        action = None
        if row["status"] == "PENDING_REVIEW" and st.button("Approve reviewed FAQ", key="approve-" + faq_id):
            action = "APPROVED"
        if row["status"] in {"GENERATED", "PENDING_REVIEW", "APPROVED"} and st.button("Reject FAQ", key="reject-" + faq_id):
            action = "REJECTED"
        if principal.role == "admin" and row["status"] == "APPROVED" and st.button("Publish approved FAQ", key="publish-" + faq_id):
            action = "PUBLISHED"
        if action:
            try:
                service.transition(faq_id, action, principal)
                st.success(f"FAQ is now {action.lower().replace('_', ' ')}.")
            except (ValueError, PermissionError) as error:
                st.warning(str(error))
