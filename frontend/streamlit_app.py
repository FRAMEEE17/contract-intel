"""Streamlit UI for contract QA, summarization, and a contract library.

Pages (sidebar nav): Library (upload + repository + clause-abstraction grid),
Chat (across all contracts or one), and Summarize. Calls the FastAPI backend.
"""
from __future__ import annotations

import os

import httpx
import streamlit as st
from streamlit_option_menu import option_menu

API_URL = os.environ.get("API_URL", "http://localhost:8000")

SAMPLE_CONTRACT = """DISTRIBUTOR AGREEMENT

This Distributor Agreement (the "Agreement") is made between Signature Ortho Pty Ltd
("Supplier") and CPM Medical Consultants ("Distributor") as of the Effective Date.

1. Appointment. The Supplier appoints the Distributor as its non-exclusive distributor
of the Products in the Territory.

2. Term. This Agreement begins on the Effective Date and continues for three (3) years.

3. Termination. Either party may terminate for material breach on thirty (30) days'
written notice if the breach is not cured within that period.

4. Confidentiality. Each party shall keep the other's Confidential Information secret.

5. Limitation of Liability. Total liability is capped at the fees paid in the prior
twelve (12) months.

6. Governing Law. This Agreement is governed by the laws of the State of New York.
"""


def _post(path: str, **kw):
    return httpx.post(f"{API_URL}{path}", timeout=180, **kw)


def _get(path: str):
    return httpx.get(f"{API_URL}{path}", timeout=60)


st.set_page_config(page_title="Contract-Intel", page_icon="📄", layout="wide")

if "active_text" not in st.session_state:
    st.session_state.active_text = SAMPLE_CONTRACT
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    page = option_menu(
        menu_title="Contract-Intel",
        options=["Library", "Chat", "Summarize"],
        icons=["folder", "chat-dots", "file-earmark-text"],
        menu_icon="clipboard-check",
        default_index=0,
    )
    st.caption(f"API {API_URL}")


# --- Library: upload -> repository, list, clause-abstraction grid ---
if page == "Library":
    st.title("📚 Contract Library")
    st.caption("Upload contracts into the searchable repository, then extract key clauses.")

    c1, c2 = st.columns(2)
    with c1:
        pdf = st.file_uploader("Add a PDF to the library", type="pdf")
        if pdf is not None and st.button("Add PDF", use_container_width=True):
            try:
                text = _post("/ingest", files={"file": (pdf.name, pdf.getvalue(), "application/pdf")}).json()["text"]
                if text.strip():
                    _post("/contracts", json={"title": pdf.name, "document_text": text})
                    st.success(f"Added {pdf.name}")
                else:
                    st.warning("No text layer found (a scanned PDF needs the OCR fallback).")
            except Exception as exc:
                st.error(f"Add failed: {exc}")
    with c2:
        if st.button("Add sample contract", use_container_width=True):
            try:
                _post("/contracts", json={"title": "Sample Distributor Agreement", "document_text": SAMPLE_CONTRACT})
                st.success("Added the sample.")
            except Exception as exc:
                st.error(f"Add failed: {exc}")

    try:
        contracts = _get("/contracts").json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        st.stop()

    if not contracts:
        st.info("The library is empty. Add a PDF or the sample above.")
    else:
        st.subheader(f"{len(contracts)} contract(s)")
        st.dataframe(contracts, use_container_width=True, hide_index=True)

        selected = st.selectbox("Extract key clauses for", [c["document_id"] for c in contracts])
        if st.button("Extract key clauses", type="primary"):
            with st.spinner("Running the clause checklist..."):
                try:
                    clauses = _post("/abstract", json={"document_id": selected}).json()["clauses"]
                except Exception as exc:
                    st.error(f"API error: {exc}")
                    st.stop()
            rows = [
                {"Clause": name,
                 "Status": "✅ present" if cell["present"] else "🔴 not specified",
                 "Detail": (cell["answer"][:90] if cell["present"] else "—")}
                for name, cell in clauses.items()
            ]
            st.table(rows)
            missing = [n for n, c in clauses.items() if not c["present"]]
            if missing:
                st.warning("Missing / not specified: " + ", ".join(missing))


# --- Chat: across all contracts or one ---
elif page == "Chat":
    st.title("💬 Contract Chat")
    try:
        contracts = _get("/contracts").json()
    except Exception:
        contracts = []
    scope = st.selectbox("Search scope", ["All contracts"] + [c["document_id"] for c in contracts])
    st.caption("Answers come from the indexed contracts, or abstain when they are silent.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        if not contracts:
            st.warning("The library is empty — add a contract on the Library page first.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching and answering..."):
                payload = {"question": prompt}
                if scope != "All contracts":
                    payload["document_id"] = scope
                try:
                    data = _post("/answer", json=payload).json()
                except Exception as exc:
                    st.error(f"API error: {exc}")
                    st.stop()
            if data["blocked"]:
                body = "⛔ Blocked by the guardrail."
            elif data["malformed"]:
                body = "⚠️ Malformed answer (counted as a failure)."
            elif data["abstained"] or data["no_context"]:
                body = "🤷 **Not specified** — the contract(s) do not address this."
            else:
                body = data["answer"]
            st.markdown(body)
            if data.get("citations"):
                with st.expander("Citations"):
                    for c in data["citations"]:
                        st.write(f"- `{c['document_id']}` · `{c['chunk_id']}` · score {c['score']:.3f}")
            st.session_state.messages.append({"role": "assistant", "content": body})


# --- Summarize: a library contract, or pasted text ---
elif page == "Summarize":
    st.title("📄 Contract Summarizer")
    try:
        contracts = _get("/contracts").json()
    except Exception:
        contracts = []

    source = st.radio("Source", ["From library", "Paste text"], horizontal=True,
                      index=0 if contracts else 1)

    payload = None
    if source == "From library":
        if not contracts:
            st.info("The library is empty — add a contract on the Library page first.")
        else:
            selected = st.selectbox("Contract", [c["document_id"] for c in contracts])
            if st.button("Summarize", type="primary"):
                payload = {"document_id": selected}
    else:
        st.session_state.active_text = st.text_area(
            "Contract text", value=st.session_state.active_text, height=260
        )
        if st.button("Summarize", type="primary"):
            if not st.session_state.active_text.strip():
                st.warning("Paste a contract first.")
                st.stop()
            payload = {"document_text": st.session_state.active_text}

    if payload is not None:
        with st.spinner("Summarizing..."):
            try:
                data = _post("/summarize", json=payload).json()
            except Exception as exc:
                st.error(f"API error: {exc}")
                st.stop()

        if data.get("title"):
            st.subheader(data["title"])
        if data.get("parties"):
            st.write("**Parties:** " + ", ".join(
                f"{p.get('name', '?')} ({p.get('role', '')})" for p in data["parties"]))
        term = data.get("term") or {}
        if term:
            st.write(f"**Term:** {term.get('initial_term', '')} "
                     f"({term.get('start_date', '?')} → {term.get('end_date', '?')})")
        if data.get("overview"):
            st.write(data["overview"])
        if data.get("key_conditions"):
            st.markdown("**Key conditions**")
            st.table([{"Priority": c.get("priority", ""), "Condition": c.get("description", ""),
                       "Impact": c.get("impact", "")} for c in data["key_conditions"]])
        if data.get("important_dates"):
            st.markdown("**Important dates**")
            st.table([{"Priority": d.get("priority", ""), "Date": d.get("date", ""),
                       "Description": d.get("description", "")} for d in data["important_dates"]])
