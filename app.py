import json
import os

import streamlit as st

from src.retrieve import load_vectorstore, retrieve_chunks
from src.generate import generate_answer
from src.generate_gemini import generate_answer_gemini, generate_gemini_for_consistency
from src.trust_score import evaluate_answer

st.set_page_config(page_title="Hallucination Detector", page_icon="🔍", layout="wide")


# ---------- Cached resources ----------
@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


# ---------- Helpers ----------
def trust_label(score):
    if score >= 0.7:
        return "🟢 High trust"
    if score >= 0.5:
        return "🟡 Medium trust"
    return "🔴 Low trust"


def overall_trust(query, answer, context, consistency_fn=None):
    details = evaluate_answer(query, answer, context, consistency_generate_fn=consistency_fn)
    if not details:
        return 1.0, details
    return min(d["trust_score"] for d in details), details


def run_llama(query, chunks, context):
    try:
        answer = generate_answer(query, chunks)
        trust, details = overall_trust(query, answer, context)
        return {"answer": answer, "trust": trust, "details": details}
    except Exception as e:
        return {"error": str(e)}


def run_gemini(query, context):
    try:
        answer = generate_answer_gemini(query, context)
        trust, details = overall_trust(query, answer, context, generate_gemini_for_consistency)
        return {"answer": answer, "trust": trust, "details": details}
    except Exception as e:
        return {"error": str(e)}


def render_model_column(title, result):
    st.subheader(title)
    if "error" in result:
        st.error(f"This model failed: {result['error']}")
        return

    trust = result["trust"]
    st.metric("Trust score", f"{trust:.3f}")
    st.progress(min(max(trust, 0.0), 1.0))
    st.markdown(f"**{trust_label(trust)}**")

    flagged = [d for d in result["details"] if d["flagged"]]
    if flagged:
        st.warning(f"Detected as possible hallucination ({len(flagged)} sentence(s) flagged)")
    else:
        st.success("Not detected as hallucinated")

    st.markdown("**Generated answer**")
    st.write(result["answer"])

    if flagged:
        st.markdown("**Flagged sentences**")
        for d in flagged:
            st.markdown(f"- 🚩 {d['sentence']}")

    with st.expander("Per-sentence breakdown"):
        st.dataframe(
            [
                {
                    "sentence": d["sentence"],
                    "similarity": round(d["similarity"], 3),
                    "judge": round(d["judge_score"], 3),
                    "consistency": round(d["consistency_score"], 3),
                    "trust": round(d["trust_score"], 3),
                    "flagged": d["flagged"],
                }
                for d in result["details"]
            ],
            use_container_width=True,
        )


def compute_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "Accuracy": round(accuracy, 4),
        "Samples": len(y_true),
    }


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🔍 Hallucination Detector")
    st.caption("RAG pipeline + 3 detectors (similarity, LLM-as-Judge, self-consistency) combined into a trust score.")
    k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=8, value=3)
    st.markdown("**Trust score legend**")
    st.markdown("🟢 ≥ 0.70 high  \n🟡 0.50–0.70 medium  \n🔴 < 0.50 possible hallucination")
    st.caption("Requires Ollama running locally and GEMINI_API_KEY set.")

tab_ask, tab_eval = st.tabs(["Ask a question", "Evaluation results"])

# ---------- Tab 1: live comparison ----------
with tab_ask:
    st.header("Llama 3 vs Gemini: live hallucination check")

    with st.form("query_form"):
        query = st.text_input("Your question", placeholder="e.g. Who designed the Statue of Liberty?")
        submitted = st.form_submit_button("Compare models")

    if submitted and query.strip():
        query = query.strip()
        with st.spinner("Retrieving context, generating answers, running detectors (this can take a minute)..."):
            vectorstore = get_vectorstore()
            chunks = retrieve_chunks(vectorstore, query, k=k)
            context = [c.page_content for c in chunks]
            llama = run_llama(query, chunks, context)
            gemini = run_gemini(query, context)

        col1, col2 = st.columns(2)
        with col1:
            render_model_column("Llama 3 (local)", llama)
        with col2:
            render_model_column("Gemini (cloud)", gemini)

        st.divider()
        candidates = {name: r for name, r in [("Llama 3", llama), ("Gemini", gemini)] if "error" not in r}
        if candidates:
            best = max(candidates, key=lambda n: candidates[n]["trust"])
            st.subheader(f"✅ Effective answer (from {best})")
            st.info(candidates[best]["answer"])
            if len(candidates) == 2:
                st.caption(
                    f"Selected the answer with the higher trust score: "
                    f"Llama 3 = {llama['trust']:.3f}, Gemini = {gemini['trust']:.3f}"
                )

        with st.expander("Retrieved context used for both models"):
            for i, c in enumerate(context, 1):
                st.markdown(f"**Chunk {i}**")
                st.write(c)

    elif submitted:
        st.warning("Please type a question first.")

# ---------- Tab 2: evaluation results ----------
with tab_eval:
    st.header("Evaluation results")

    st.subheader("RAGTruth evaluation (trust-score system)")
    rt = load_json("eval_results.json")
    if rt:
        valid = [r for r in rt if r.get("predicted_label") is not None]
        m = compute_metrics([r["true_label"] for r in valid], [r["predicted_label"] for r in valid])
        st.table([m])
    else:
        st.info("eval_results.json not found in the project root.")

    st.subheader("Llama 3 vs Gemini (hallucination detection performance)")
    cmp_results = load_json("eval_compare_results.json")
    if cmp_results:
        rows = []
        for key, label in [("llama", "Llama 3"), ("gemini", "Gemini")]:
            valid = [r for r in cmp_results if r.get(key, {}).get("predicted_hallucinated") is not None]
            m = compute_metrics(
                [r["true_label"] for r in valid],
                [r[key]["predicted_hallucinated"] for r in valid],
            )
            rows.append({"Model": label, **m})
        st.table(rows)
    else:
        st.info("eval_compare_results.json not found in the project root.")