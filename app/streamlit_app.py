from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from config import (
    ARABIC_STOPWORDS_PATH,
    DOCUMENTS_PATH,
    LEGAL_STOPWORDS_PATH,
    VECTOR_STORE_PATH,
)
from src.morphology.analyzer import ArabicMorphologyAnalyzer
from src.rag.generator import ExtractiveGenerator
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore
from src.search.engine import SearchEngine
from src.utils.io import load_json

st.set_page_config(page_title="البحث في القرارات العقارية", layout="wide")
st.markdown(
    "<style>.stApp { direction: rtl; text-align: right; }</style>",
    unsafe_allow_html=True,
)
st.title("البحث في القرارات العقارية")

if not DOCUMENTS_PATH.exists():
    st.info("شغّل scripts/01_extract.py بعد إضافة ملفات القرارات.")
    st.stop()
documents = load_json(DOCUMENTS_PATH)
if not documents:
    st.info("لا توجد وثائق مفهرسة حتى الآن.")
    st.stop()

analyzer = ArabicMorphologyAnalyzer([ARABIC_STOPWORDS_PATH, LEGAL_STOPWORDS_PATH])
tab_search, tab_rag = st.tabs(["البحث", "اسأل الوثائق"])
with tab_search:
    query = st.text_input("عبارة البحث")
    if query:
        for result in SearchEngine(documents, analyzer).search(query):
            st.markdown(f"**{result['doc_id']}** — {result['score']:.3f}")
            st.write(result["document"]["text"][:1500])
with tab_rag:
    question = st.text_input("اكتب سؤالك", key="rag_question")
    if not VECTOR_STORE_PATH.exists():
        st.info("شغّل scripts/08_build_rag.py لإنشاء فهرس المقاطع.")
    elif question:
        terms = {doc["doc_id"]: analyzer.analyze(doc["text"]) for doc in documents}
        store = VectorStore.load(VECTOR_STORE_PATH)
        retriever = HybridRetriever(terms, store)
        retrieved = retriever.retrieve(question, analyzer.analyze(question))
        passages = [item["chunk"] for item in retrieved if item["chunk"]]
        answer = ExtractiveGenerator().generate(question, passages)
        st.write(answer["answer"])
        if answer["sources"]:
            st.caption("المصادر: " + "، ".join(answer["sources"]))
