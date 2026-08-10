from __future__ import annotations

import os

import httpx
import streamlit as st # type: ignore

API_URL = os.environ.get("API_URL", "http://localhost:8000")

SAMPLE_CONTRACT = """DISTRIBUTOR AGREEMENT

1. Governing Law. This Agreement shall be governed by and construed in accordance
with the laws of the State of New York, without regard to its conflict-of-laws rules.

2. Term. This Agreement begins on the Effective Date and continues for three (3)
years, unless terminated earlier under Section 3.

3. Termination. Either party may terminate this Agreement for material breach on
thirty (30) days' written notice if the breach remains uncured.

4. Confidentiality. Each party shall keep the other's Confidential Information
secret and use it only to perform under this Agreement.
"""

st.set_page_config(page_title="Contract-Intel", page_icon="📄")
st.title("📄 Contract-Intel — Clause QA")
st.caption("Ask about a contract. The system answers from the text, or abstains when the contract is silent.")

with st.sidebar:
    st.subheader("Contract")
    document_text = st.text_area("Contract text", value=SAMPLE_CONTRACT, height=340)
    st.caption(f"API: {API_URL}")

question = st.text_input("Question", value="Which state's law governs this agreement?")

if st.button("Ask", type="primary"):
    if not document_text.strip() or not question.strip():
        st.warning("Provide both a contract and a question.")
        st.stop()
    with st.spinner("Retrieving and answering..."):
        try:
            resp = httpx.post(
                f"{API_URL}/answer",
                json={"question": question, "document_text": document_text},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            st.error(f"API error: {exc}")
            st.stop()

    if data["blocked"]: # type: ignore
        st.error("Blocked by the guardrail.")
    elif data["malformed"]: # type: ignore
        st.error("The model returned a malformed answer (counted as a failure, not shown).")
    elif data["abstained"]: # type: ignore
        st.info("**Not specified** — the contract does not address this question.")
    else:
        st.success(data["answer"]) # type: ignore

    if data.get("citations"): # type: ignore
        st.subheader("Citations")
        for c in data["citations"]: # type: ignore
            st.write(f"- `{c['chunk_id']}` · score {c['score']:.3f} · section: {c.get('section') or 'n/a'}")
