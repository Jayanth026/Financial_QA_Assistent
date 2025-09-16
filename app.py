import os
import io
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber

# Ollama integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    import requests

st.set_page_config(page_title="Financial Document Q&A Assistant", layout="wide")

def safe_float(x):
    try:
        return float(re.sub(r"[^0-9.\-]", "", str(x)))
    except Exception:
        return None

def ensure_ollama_running():
    try:
        if OLLAMA_AVAILABLE:
            _ = ollama.list()
            return True
        else:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            return r.status_code == 200
    except Exception:
        return False

def ollama_embed(texts, model="nomic-embed-text"):
    if OLLAMA_AVAILABLE:
        return np.vstack([np.array(ollama.embeddings(model=model, prompt=t)["embedding"], dtype="float32") for t in texts])
    else:
        res = []
        for t in texts:
            r = requests.post("http://localhost:11434/api/embeddings",
                              json={"model": model, "prompt": t})
            res.append(np.array(r.json()["embedding"], dtype="float32"))
        return np.vstack(res)

def ollama_chat(system_prompt, user_prompt, model="mistral:7b-instruct"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if OLLAMA_AVAILABLE:
        res = ollama.chat(model=model, messages=messages)
        return res["message"]["content"]
    else:
        r = requests.post("http://localhost:11434/api/chat",
                          json={"model": model, "messages": messages})
        return r.json()["message"]["content"]

def load_pdf(file_bytes, filename):
    texts, tables = [], []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            if txt.strip():
                texts.append({"filename": filename, "page": i, "text": txt})
            for t in page.extract_tables() or []:
                df = pd.DataFrame(t)
                tables.append({"filename": filename, "page": i, "df": df})
    return texts, tables

def load_excel(file_bytes, filename):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    texts, tables = [], []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        tables.append({"filename": filename, "sheet": sheet, "df": df})
    return texts, tables

def chunk_text(text, max_chars=1000, overlap=200):
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if start >= n: break
    return chunks

def build_corpus_and_embeddings(all_texts):
    corpus, meta = [], []
    for record in all_texts:
        for ch in chunk_text(record["text"]):
            corpus.append(ch)
            meta.append({k: v for k, v in record.items() if k != "text"})
    embeddings = ollama_embed(corpus, "nomic-embed-text") if corpus else None
    return corpus, meta, embeddings

def retrieve(query, corpus, meta, embeddings, top_k=5):
    if embeddings is None or not corpus: return []
    q_emb = ollama_embed([query], "nomic-embed-text")[0]
    sims = (embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)) @ (q_emb / np.linalg.norm(q_emb))
    idxs = np.argsort(-sims)[:top_k]
    return [{"chunk": corpus[i], "meta": meta[i], "score": float(sims[i])} for i in idxs]

METRIC_SYNONYMS = {
    "revenue": ["revenue", "sales", "net sales"],
    "net income": ["net income", "profit", "net profit"],
    "operating expenses": ["operating expenses", "expenses", "opex"],
    "cash from operations": ["operating activities", "cash flow from operating activities"],
    "total assets": ["total assets"],
    "total liabilities": ["total liabilities"],
    "equity": ["equity", "shareholders' equity"]
}

def extract_metric_from_tables(question, tables):
    q = question.lower()
    target = None
    for metric, syns in METRIC_SYNONYMS.items():
        if any(s in q for s in syns):
            target = metric
            break
    if not target:
        return None

    year_match = re.findall(r"(20\d{2})", q)
    year_filter = year_match[0] if year_match else None

    best_hit = None
    for t in tables:
        df = t["df"]
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        first_col = df.columns[0]
        series = df[first_col].astype(str).str.lower()
        matches = series.apply(lambda x: any(s in x for s in METRIC_SYNONYMS[target]))
        if matches.any():
            row = df[matches].iloc[0]

            # Case 1: Year specified → pick that column
            if year_filter and year_filter in df.columns.astype(str):
                val = safe_float(row[year_filter])
                if val is not None:
                    best_hit = {"metric": target, "year": year_filter, "value": val}
                    break

            numeric_vals = []
            for c in df.columns[1:]:
                val = safe_float(row[c])
                if val is not None:
                    numeric_vals.append(val)
            if numeric_vals:
                best_hit = {"metric": target, "year": "all", "value": sum(numeric_vals)}

    return best_hit

if "texts" not in st.session_state:
    st.session_state.update({"texts": [], "tables": [], "corpus": [], "meta": [], "embeddings": None, "chat": []})

st.title("Financial Document Q&A Assistant")

uploaded_files = st.sidebar.file_uploader("Upload PDFs/Excels", type=["pdf", "xlsx"], accept_multiple_files=True)
if st.sidebar.button("Clear"):
    st.session_state.update({"texts": [], "tables": [], "corpus": [], "meta": [], "embeddings": None, "chat": []})
    st.rerun()

if uploaded_files:
    for uf in uploaded_files:
        data = uf.read()
        if uf.name.endswith(".pdf"):
            t_texts, t_tables = load_pdf(data, uf.name)
        else:
            t_texts, t_tables = load_excel(data, uf.name)
        st.session_state["texts"] += t_texts
        st.session_state["tables"] += t_tables
    st.success("Files loaded!")

    # Clearer Table Preview
    st.markdown("---")
    st.subheader("Preview of Extracted Tables")
    if st.session_state["tables"]:
        for i, t in enumerate(st.session_state["tables"], start=1):
            meta = {k: v for k, v in t.items() if k != "df"}
            st.caption(f"Table {i} — source: {json.dumps(meta)}")
            st.dataframe(t["df"].head(10), use_container_width=True)
    else:
        st.info("No tables detected. If your PDF is scanned, consider OCR.")

if st.sidebar.button("Build Index"):
    if not ensure_ollama_running():
        st.error("Ollama not running. Start it with models pulled.")
    else:
        st.session_state["corpus"], st.session_state["meta"], st.session_state["embeddings"] = build_corpus_and_embeddings(st.session_state["texts"])
        st.success("Index built!")

prompt = st.chat_input("Ask about revenue, net income, cash flow, assets...")
for role, content in st.session_state["chat"]:
    with st.chat_message(role):
        st.markdown(content)

if prompt:
    with st.chat_message("user"): st.markdown(prompt)
    st.session_state["chat"].append(("user", prompt))

    metric_hit = extract_metric_from_tables(prompt, st.session_state["tables"])
    retrieved = retrieve(prompt, st.session_state["corpus"], st.session_state["meta"], st.session_state["embeddings"])
    context = "\n".join(r["chunk"] for r in retrieved)

    system_prompt = "You are a financial assistant. Use CONTEXT + STRUCTURED METRIC to answer precisely."
    user_prompt = f"QUESTION: {prompt}\n\nCONTEXT:\n{context}\n\nSTRUCTURED_METRIC:\n{metric_hit}"

    answer = "I couldn't reach the model."
    try:
        answer = ollama_chat(system_prompt, user_prompt, model="mistral:7b-instruct")
    except Exception as e:
        st.error("LLM error: " + str(e))

    with st.chat_message("assistant"):
        if metric_hit:
            if metric_hit["year"] == "all":
                st.markdown(f"**{metric_hit['metric'].title()} (all years): {metric_hit['value']:,}**")
            else:
                st.markdown(f"**{metric_hit['metric'].title()} ({metric_hit['year']}): {metric_hit['value']:,}**")
        st.markdown(answer)

    st.session_state["chat"].append(("assistant", answer))

    # Show transparency panels
    if metric_hit:
        with st.expander("Structured metric (from parsed table)"):
            st.json(metric_hit)
    if retrieved:
        with st.expander("Top retrieved text chunks"):
            for r in retrieved:
                st.markdown(f"**Score:** {r['score']:.3f} | **Meta:** `{r['meta']}`")
                st.code(r["chunk"][:1000])
