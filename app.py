import streamlit as st
from search_engine import semantic_search

st.title("📘 Semantic Document Search (FAISS + SBERT)")
st.write("Upload PDFs, DOCX, TXT inside **documents/** and run `ingest.py` first.")

query = st.text_input("🔍 Enter your question:")

top_k = st.slider("Results", 1, 5, 3)

if st.button("Search"):
    if not query.strip():
        st.error("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            results = semantic_search(query, top_k=top_k)

        if len(results) == 0:
            st.warning("No strong matches found.")
        else:
            for r in results:
                st.subheader(f"📄 {r['source']}")
                st.write(f"**Similarity Score:** {r['score']:.4f}")
                st.info(r['text'])
                st.markdown("---")
