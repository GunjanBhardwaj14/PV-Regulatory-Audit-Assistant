import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

st.set_page_config(page_title="PV Regulatory Audit Assistant", layout="wide")
st.title("Regulatory Compliance & Audit Assistant")
st.caption("Powered by EMA GVP Module VI (Rev 2) & Biomedical RAG")


@st.cache_resource
def load_vector_db():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )


vector_db = load_vector_db()

uploaded_file = st.file_uploader("Upload a Draft Safety Report (.txt)", type=["txt"])

default_report = """Draft Safety Report #104
Reporter: Dr. Meera Nair, Cardiologist (meera.nair@clinic.in)
Patient: Details not recorded yet.
Suspect Drug: Atorvastatin 20mg daily
Adverse Reaction: Severe muscle pain and dark urine reported 10 days after starting therapy."""

if uploaded_file is not None:
    report_text = uploaded_file.read().decode("utf-8")
else:
    report_text = default_report

draft_report = st.text_area("Draft Safety Report to Audit", value=report_text, height=180)

audit_focus = st.text_input(
    "Compliance Rule / Audit Focus",
    value="Four minimum criteria required for ICSR validation: identifiable reporter, identifiable patient, suspect medicinal product, suspected adverse reaction"
)

if st.button("Run Compliance Audit", type="primary"):
    if not draft_report.strip():
        st.warning("Please upload or paste a draft safety report first.")
    else:
        with st.spinner("Cross-referencing draft against EMA GVP Module VI..."):
            results = vector_db.similarity_search(audit_focus, k=5)

            context = "\n\n".join(
                [f"[PDF Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}" for doc in results]
            )

            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

            messages = [
                (
                    "system",
                    "You are a strict Pharmacovigilance Regulatory Compliance Auditor. "
                    "Audit the Draft Safety Report using ONLY the EMA GVP Module VI rules provided. "
                    "State clearly whether the report is COMPLIANT or NON-COMPLIANT, list any missing mandatory elements, "
                    "and cite the exact PDF page numbers from the provided rules. Do not invent rules."
                ),
                (
                    "human",
                    f"EMA GVP MODULE VI RULES:\n{context}\n\n"
                    f"DRAFT SAFETY REPORT:\n{draft_report}\n\n"
                    "Perform the compliance audit now."
                )
            ]

            response = llm.invoke(messages)

        st.subheader("Audit Verdict")
        st.markdown(response.text)

        with st.expander("View Retrieved EMA GVP Module VI Sources"):
            for i, doc in enumerate(results, start=1):
                page_num = doc.metadata.get("page", 0) + 1
                st.markdown(f"**Source {i} — PDF Page {page_num}**")
                st.write(doc.page_content)
                st.divider()