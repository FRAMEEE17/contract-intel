"""Streamlit UI for contract-clause QA + summarization. Calls the FastAPI backend.

Pages (sidebar nav): Summarize and Chat. The active contract is shared in session
state and can come from a PDF upload (/ingest), pasted text, or the bundled sample.
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

2. Territory. The Territory is the United States.

3. Term. This Agreement begins on the Effective Date and continues for three (3) years,
and renews for successive one-year terms unless either party gives 60 days' notice.

4. Obligations of the Distributor. The Distributor shall use commercially reasonable
efforts to market and sell the Products and shall not sell competing products.

5. Prices and Payment. The Distributor shall pay each invoice within thirty (30) days.
Late amounts accrue interest at 1.5% per month.

6. Confidentiality. Each party shall keep the other's Confidential Information secret
and use it only to perform under this Agreement, for three years after termination.

7. Intellectual Property. The Supplier retains all rights in the Products and its
trademarks. The Distributor is granted no license except as needed to resell.

8. Limitation of Liability. Neither party is liable for indirect or consequential
damages. Total liability is capped at the fees paid in the prior twelve (12) months.

9. Termination. Either party may terminate for material breach on thirty (30) days'
written notice if the breach is not cured within that period.

10. Governing Law. This Agreement is governed by the laws of the State of New York,
without regard to its conflict-of-laws principles.
"""

st.set_page_config(page_title="Contract-Intel", page_icon="📄", layout="wide")

if "contract_text" not in st.session_state:
    st.session_state.contract_text = SAMPLE_CONTRACT
if "messages" not in st.session_state:
    st.session_state.messages = []


def _load_pdf(file) -> None:
    with st.spinner("Extracting text from PDF..."):
        try:
            resp = httpx.post(
                f"{API_URL}/ingest",
                files={"file": (file.name, file.getvalue(), "application/pdf")},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            st.error(f"Ingest failed: {exc}")
            return
    if not data["text"].strip():
        st.warning("No text layer found (a scanned PDF needs the Azure OCR fallback).")
        return
    st.session_state.contract_text = data["text"]
    st.success(f"Loaded {data['chars']} chars across {data['pages']} page(s).")


# --- sidebar: navigation + contract source ---
with st.sidebar:
    page = option_menu(
        menu_title="Contract-Intel",
        options=["Summarize", "Chat"],
        icons=["file-earmark-text", "chat-dots"],
        menu_icon="clipboard-check",
        default_index=0,
    )
    st.divider()
    st.subheader("Contract")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded is not None and st.button("Extract PDF"):
        _load_pdf(uploaded)
    if st.button("Load sample"):
        st.session_state.contract_text = SAMPLE_CONTRACT
    st.session_state.contract_text = st.text_area(
        "Contract text", value=st.session_state.contract_text, height=260
    )
    st.caption(f"{len(st.session_state.contract_text)} chars · API {API_URL}")


def _require_contract() -> bool:
    if not st.session_state.contract_text.strip():
        st.warning("Load or paste a contract first (sidebar).")
        return False
    return True


# --- Summarize page ---
if page == "Summarize":
    st.title("📄 Contract Summarizer")
    st.caption("A plain-English summary of the active contract.")
    if st.button("Summarize", type="primary") and _require_contract():
        with st.spinner("Summarizing..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/summarize",
                    json={"document_text": st.session_state.contract_text},
                    timeout=120,
                )
                resp.raise_for_status()
                st.markdown(resp.json()["summary"])
            except Exception as exc:
                st.error(f"API error: {exc}")

# --- Chat page ---
elif page == "Chat":
    st.title("💬 Contract Chat")
    st.caption("Ask about the contract. Answers come from the text, or abstain when it is silent.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the contract..."):
        if not _require_contract():
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and answering..."):
                try:
                    resp = httpx.post(
                        f"{API_URL}/answer",
                        json={"question": prompt, "document_text": st.session_state.contract_text},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    st.error(f"API error: {exc}")
                    st.stop()

            if data["blocked"]:
                body = "⛔ Blocked by the guardrail."
            elif data["malformed"]:
                body = "⚠️ The model returned a malformed answer (counted as a failure)."
            elif data["abstained"]:
                body = "🤷 **Not specified** — the contract does not address this."
            else:
                body = data["answer"]
            st.markdown(body)

            if data.get("citations"):
                with st.expander("Citations"):
                    for c in data["citations"]:
                        st.write(f"- `{c['chunk_id']}` · score {c['score']:.3f} · section: {c.get('section') or 'n/a'}")
            st.session_state.messages.append({"role": "assistant", "content": body})
