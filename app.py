import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from src.retrieve import load_vectorstore, retrieve_chunks
from src.generate import generate_answer
from src.generate_gemini import generate_answer_gemini, generate_gemini_for_consistency
from src.detectors.similarity import sentence_similarity_scores, split_into_sentences
from src.detectors.judge import judge_sentences, get_judge_score
from src.detectors.consistency import check_self_consistency
from src.trust_score import is_refusal

st.set_page_config(page_title="Hallucination Detector", page_icon="🔍", layout="wide")


# =====================================================================
# Cached resources
# =====================================================================
@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


# =====================================================================
# Core logic: analysis + scoring
# =====================================================================
def analyze_answer(query, answer, context, consistency_fn=None):
    """Run all 3 detectors once and keep the RAW component scores,
    so trust can be recomputed instantly with any weights/threshold."""
    sim = sentence_similarity_scores(answer, context)
    judge = judge_sentences(answer, context)
    cons = check_self_consistency(query, context, generate_fn=consistency_fn)

    rows = []
    for s, j in zip(sim, judge):
        rows.append({
            "sentence": s["sentence"],
            "similarity": s["similarity"],
            "judge": j["judge_score"],
            "consistency": cons["consistency_score"],
            "refusal": is_refusal(s["sentence"]),
        })
    return {
        "answer": answer,
        "rows": rows,
        "consistency_score": cons["consistency_score"],
        "consistency_answers": cons["answers"],
    }


def score_rows(rows, weights, threshold):
    ws, wj, wc = weights
    out = []
    for r in rows:
        if r["refusal"]:
            trust = 1.0
        else:
            trust = ws * r["similarity"] + wj * r["judge"] + wc * r["consistency"]
        out.append({
            "Sentence": r["sentence"],
            "Similarity": round(r["similarity"], 3),
            "Judge": round(r["judge"], 3),
            "Consistency": round(r["consistency"], 3),
            "Trust": round(trust, 3),
            "Refusal": r["refusal"],
            "Flagged": trust < threshold,
        })
    df = pd.DataFrame(out)
    overall = float(df["Trust"].min()) if not df.empty else 1.0
    return df, overall


def run_llama(query, chunks, context):
    try:
        answer = generate_answer(query, chunks)
        return analyze_answer(query, answer, context)
    except Exception as e:
        return {"error": str(e)}


def run_gemini(query, context):
    try:
        answer = generate_answer_gemini(query, context)
        return analyze_answer(query, answer, context, generate_gemini_for_consistency)
    except Exception as e:
        return {"error": str(e)}


# =====================================================================
# Metrics helpers
# =====================================================================
def confusion(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    return tp, fp, fn, tn


def metrics_from(y_true, y_pred):
    tp, fp, fn, tn = confusion(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "Precision": precision, "Recall": recall, "F1": f1, "Accuracy": accuracy,
        "Samples": len(y_true), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def predict(scores, threshold):
    return [1 if s < threshold else 0 for s in scores]


def sweep(y_true, scores):
    rows = []
    for t in [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]:
        m = metrics_from(y_true, predict(scores, t))
        rows.append({
            "Threshold": t,
            "Precision": round(m["Precision"], 3),
            "Recall": round(m["Recall"], 3),
            "F1": round(m["F1"], 3),
            "Accuracy": round(m["Accuracy"], 3),
        })
    return pd.DataFrame(rows)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def show_metric_cards(m):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", f"{m['Precision']:.3f}")
    c2.metric("Recall", f"{m['Recall']:.3f}")
    c3.metric("F1-score", f"{m['F1']:.3f}")
    c4.metric("Accuracy", f"{m['Accuracy']:.3f}")
    c5.metric("Samples", m["Samples"])


def show_confusion(m):
    st.dataframe(
        pd.DataFrame(
            [[m["TN"], m["FP"]], [m["FN"], m["TP"]]],
            index=["Actually NOT hallucinated", "Actually hallucinated"],
            columns=["Predicted NOT hallucinated", "Predicted hallucinated"],
        )
    )


# =====================================================================
# Sidebar (global controls)
# =====================================================================
with st.sidebar:
    st.title("🔍 Hallucination Detector")
    st.caption("RAG pipeline + 3 detectors combined into a trust score.")

    k = st.slider("Chunks to retrieve (top-k)", 1, 8, 3)
    threshold = st.slider("Hallucination threshold", 0.10, 0.90, 0.50, 0.05,
                          help="An answer is flagged if its lowest sentence trust is below this.")

    st.markdown("**Detector weights** (auto-normalized to sum to 1)")
    w_sim = st.slider("Similarity", 0.0, 1.0, 0.3, 0.05)
    w_judge = st.slider("LLM-as-Judge", 0.0, 1.0, 0.5, 0.05)
    w_cons = st.slider("Self-consistency", 0.0, 1.0, 0.2, 0.05)
    total_w = w_sim + w_judge + w_cons
    if total_w == 0:
        WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
    else:
        WEIGHTS = (w_sim / total_w, w_judge / total_w, w_cons / total_w)
    st.caption(f"Effective weights: sim {WEIGHTS[0]:.2f} / judge {WEIGHTS[1]:.2f} / consistency {WEIGHTS[2]:.2f}")
    st.caption("Needs Ollama running and GEMINI_API_KEY set.")


tab_live, tab_lab, tab_rt, tab_cmp, tab_how = st.tabs([
    "Live Check", "Detector Lab", "RAGTruth Results", "Llama vs Gemini Results", "How It Works"
])


# =====================================================================
# Tab 1: Live Check
# =====================================================================
def render_model(title, res):
    st.subheader(title)
    if "error" in res:
        st.error(f"This model failed: {res['error']}")
        return

    df, overall = score_rows(res["rows"], WEIGHTS, threshold)
    hallucinated = overall < threshold

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall trust (lowest sentence)", f"{overall:.3f}")
    c2.metric("Self-consistency", f"{res['consistency_score']:.3f}")
    n_flag = int(df["Flagged"].sum()) if not df.empty else 0
    c3.metric("Sentences flagged", f"{n_flag} / {len(df)}")
    st.progress(min(max(overall, 0.0), 1.0))

    if hallucinated:
        st.error("🚩 Detected as HALLUCINATED")
    else:
        st.success("✅ Not detected as hallucinated")

    st.markdown("**Generated answer**")
    st.write(res["answer"])

    st.markdown("**Per-sentence scores**")
    if df.empty:
        st.info("No sentences to score.")
    else:
        st.dataframe(
            df,
            column_config={
                "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.3f"),
                "Judge": st.column_config.ProgressColumn("Judge", min_value=0, max_value=1, format="%.3f"),
                "Consistency": st.column_config.ProgressColumn("Consistency", min_value=0, max_value=1, format="%.3f"),
                "Trust": st.column_config.ProgressColumn("Trust", min_value=0, max_value=1, format="%.3f"),
            },
        )

    with st.expander("Score math (how each trust score was computed)"):
        ws, wj, wc = WEIGHTS
        st.caption(f"trust = {ws:.2f} × similarity + {wj:.2f} × judge + {wc:.2f} × consistency "
                   "(refusal sentences are fixed at 1.0)")
        for r, (_, row) in zip(res["rows"], df.iterrows()):
            if r["refusal"]:
                st.markdown(f"- *{r['sentence']}* → refusal → **1.000**")
            else:
                st.markdown(
                    f"- *{r['sentence']}*  \n"
                    f"  {ws:.2f}×{r['similarity']:.3f} + {wj:.2f}×{r['judge']:.3f} + "
                    f"{wc:.2f}×{r['consistency']:.3f} = **{row['Trust']:.3f}**"
                )

    with st.expander("Self-consistency regenerations"):
        for i, a in enumerate(res["consistency_answers"], 1):
            st.markdown(f"**Regeneration {i}**")
            st.write(a)


with tab_live:
    st.header("Llama 3 vs Gemini: live hallucination check")

    with st.form("query_form"):
        query = st.text_input("Your question", placeholder="e.g. Who designed the Statue of Liberty?")
        submitted = st.form_submit_button("Compare models")

    if submitted and query.strip():
        q = query.strip()
        with st.spinner("Retrieving context, generating answers, running all 3 detectors (about a minute)..."):
            vs = get_vectorstore()
            chunks = retrieve_chunks(vs, q, k=k)
            context = [c.page_content for c in chunks]
            st.session_state["live"] = {
                "query": q,
                "context": context,
                "llama": run_llama(q, chunks, context),
                "gemini": run_gemini(q, context),
            }
    elif submitted:
        st.warning("Please type a question first.")

    live = st.session_state.get("live")
    if live:
        st.markdown(f"**Question:** {live['query']}")
        col1, col2 = st.columns(2)
        with col1:
            render_model("Llama 3 (local)", live["llama"])
        with col2:
            render_model("Gemini (cloud)", live["gemini"])

        st.divider()
        candidates = {}
        for name, key in [("Llama 3", "llama"), ("Gemini", "gemini")]:
            res = live[key]
            if "error" not in res:
                candidates[name] = (score_rows(res["rows"], WEIGHTS, threshold)[1], res["answer"])
        if candidates:
            best = max(candidates, key=lambda n: candidates[n][0])
            st.subheader(f"✅ Effective answer (from {best})")
            st.info(candidates[best][1])
            st.caption("Chosen as the answer with the higher overall trust score: " +
                       ", ".join(f"{n} = {v[0]:.3f}" for n, v in candidates.items()))

        with st.expander("Retrieved context used for both models"):
            for i, c in enumerate(live["context"], 1):
                st.markdown(f"**Chunk {i}**")
                st.write(c)


# =====================================================================
# Tab 2: Detector Lab (run each detector on its own)
# =====================================================================
with tab_lab:
    st.header("Detector Lab")
    st.caption("Run any single detector on your own text. Useful for explaining each detector separately.")

    lab_context = st.text_area("Context (the source text)", height=140,
                               value="The Statue of Liberty is made of copper. It was a gift from France, completed in 1886.")
    lab_answer = st.text_area("Answer / sentence(s) to check", height=100,
                              value="The Statue of Liberty is made of copper. It was designed by Napoleon Bonaparte himself.")
    lab_query = st.text_input("Question (used by self-consistency)",
                              value="What is the Statue of Liberty made of, and who designed it?")

    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown("**1. Similarity**")
        if st.button("Run similarity"):
            res = sentence_similarity_scores(lab_answer, [lab_context])
            st.dataframe(pd.DataFrame(res).rename(columns={"sentence": "Sentence", "similarity": "Cosine similarity"}))
            st.caption("Cosine similarity between each sentence and the context (MiniLM embeddings).")

    with b2:
        st.markdown("**2. LLM-as-Judge**")
        if st.button("Run judge"):
            with st.spinner("Asking Llama 3 to judge each sentence..."):
                rows = [{"Sentence": s, "Judge score": get_judge_score(s, lab_context)}
                        for s in split_into_sentences(lab_answer)]
            st.dataframe(pd.DataFrame(rows))
            st.caption("0 = unsupported/contradicted, 0.5 = partial, 1 = fully supported.")

    with b3:
        st.markdown("**3. Self-consistency**")
        cons_model = st.radio("Regenerate with", ["Llama 3", "Gemini"], horizontal=True)
        if st.button("Run consistency"):
            fn = generate_gemini_for_consistency if cons_model == "Gemini" else None
            try:
                with st.spinner("Regenerating 3 times..."):
                    res = check_self_consistency(lab_query, [lab_context], generate_fn=fn)
                st.metric("Consistency score", f"{res['consistency_score']:.3f}")
                for i, a in enumerate(res["answers"], 1):
                    st.markdown(f"**Regeneration {i}:** {a}")
            except Exception as e:
                st.error(f"Failed: {e}")


# =====================================================================
# Tab 3: RAGTruth results
# =====================================================================
with tab_rt:
    st.header("RAGTruth evaluation (trust-score system)")
    rt = load_json("eval_results.json")
    if not rt:
        st.info("eval_results.json not found in the project root.")
    else:
        valid = [r for r in rt if r.get("min_trust_score") is not None]
        y_true = [r["true_label"] for r in valid]
        scores = [r["min_trust_score"] for r in valid]

        st.caption(f"{len(rt)} examples in file, {len(valid)} scored successfully, "
                   f"{len(rt) - len(valid)} skipped due to errors.")

        st.subheader(f"Metrics at threshold = {threshold:.2f} (set in sidebar)")
        m = metrics_from(y_true, predict(scores, threshold))
        show_metric_cards(m)
        st.markdown("**Confusion matrix**")
        show_confusion(m)

        st.subheader("Threshold sweep")
        sw = sweep(y_true, scores)
        st.dataframe(sw)
        st.line_chart(sw.set_index("Threshold")[["Precision", "Recall", "F1"]])
        st.download_button("Download sweep as CSV", sw.to_csv(index=False), "ragtruth_sweep.csv", "text/csv")
        st.caption("Note: picking the best threshold on the same data you report on is tuning on the test set. "
                   "For the paper, report the fixed 0.5 threshold, or tune on the training split.")

        st.subheader("Trust score distribution")
        bins = np.linspace(0, 1, 11)
        h0, _ = np.histogram([s for s, t in zip(scores, y_true) if t == 0], bins=bins)
        h1, _ = np.histogram([s for s, t in zip(scores, y_true) if t == 1], bins=bins)
        hist = pd.DataFrame(
            {"Not hallucinated": h0, "Hallucinated": h1},
            index=[f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(10)],
        )
        st.bar_chart(hist)
        st.caption("Good separation means the two groups sit in different score ranges.")


# =====================================================================
# Tab 4: Llama vs Gemini results
# =====================================================================
with tab_cmp:
    st.header("Llama 3 vs Gemini: hallucination detection performance")
    cmp = load_json("eval_compare_results.json")
    if not cmp:
        st.info("eval_compare_results.json not found in the project root.")
    else:
        data = {}
        for key, label in [("llama", "Llama 3"), ("gemini", "Gemini")]:
            valid = [r for r in cmp if r.get(key, {}).get("min_trust_score") is not None]
            data[label] = {
                "ids": [r["id"] for r in valid],
                "y_true": [r["true_label"] for r in valid],
                "scores": [r[key]["min_trust_score"] for r in valid],
            }

        st.caption(f"{len(cmp)} examples in file. Scored successfully: "
                   + ", ".join(f"{lbl} = {len(d['scores'])}" for lbl, d in data.items()))

        st.subheader(f"Metrics at threshold = {threshold:.2f} (set in sidebar)")
        rows = []
        mets = {}
        for label, d in data.items():
            m = metrics_from(d["y_true"], predict(d["scores"], threshold))
            mets[label] = m
            avg_trust = float(np.mean(d["scores"])) if d["scores"] else 0.0
            rows.append({
                "Model": label,
                "Precision": round(m["Precision"], 3),
                "Recall": round(m["Recall"], 3),
                "F1": round(m["F1"], 3),
                "Accuracy": round(m["Accuracy"], 3),
                "Examples tested": m["Samples"],
                "Avg trust score": round(avg_trust, 3),
            })
        st.dataframe(pd.DataFrame(rows))

        cA, cB = st.columns(2)
        for col, label in zip((cA, cB), data.keys()):
            with col:
                st.markdown(f"**{label}: confusion matrix**")
                show_confusion(mets[label])

        st.subheader("F1 vs threshold")
        f1_df = pd.DataFrame({"Threshold": sweep([0], [0.5])["Threshold"]})
        for label, d in data.items():
            f1_df[label] = sweep(d["y_true"], d["scores"])["F1"]
        st.line_chart(f1_df.set_index("Threshold"))

        st.subheader("Full sweep per model")
        for label, d in data.items():
            with st.expander(f"{label} threshold sweep"):
                sw = sweep(d["y_true"], d["scores"])
                st.dataframe(sw)
                st.download_button(f"Download {label} sweep", sw.to_csv(index=False),
                                   f"{label.replace(' ', '_').lower()}_sweep.csv", "text/csv",
                                   key=f"dl_{label}")

        st.subheader("Per-example results")
        table = []
        for r in cmp:
            lm = r.get("llama", {}).get("min_trust_score")
            gm = r.get("gemini", {}).get("min_trust_score")
            table.append({
                "id": r["id"],
                "Actual (1 = hallucinated)": r["true_label"],
                "Llama trust": None if lm is None else round(lm, 3),
                "Llama predicted": None if lm is None else int(lm < threshold),
                "Gemini trust": None if gm is None else round(gm, 3),
                "Gemini predicted": None if gm is None else int(gm < threshold),
            })
        st.dataframe(pd.DataFrame(table))


# =====================================================================
# Tab 5: How it works
# =====================================================================
with tab_how:
    st.header("How the system works")
    st.markdown(f"""
**Pipeline:** question → retrieve context (ChromaDB, MiniLM embeddings) → generate answer (Llama 3 / Gemini)
→ 3 detectors score every sentence → weighted trust score → verdict.

| Detector | Question it asks | How |
|---|---|---|
| **Similarity** | Is this sentence on-topic with the context? | Cosine similarity of sentence vs context embeddings |
| **LLM-as-Judge** | Is this claim supported by the context? | Llama 3 rates support from 0 to 1 |
| **Self-consistency** | Does the model repeat itself? | Regenerate 3 times, average pairwise similarity |

**Trust score per sentence:**
`trust = {WEIGHTS[0]:.2f} × similarity + {WEIGHTS[1]:.2f} × judge + {WEIGHTS[2]:.2f} × consistency`

**Rules on top of the formula**
- Honest refusals ("I don't know based on the context") are fixed at trust = 1.0.
- Answer-level trust = the lowest sentence trust (one bad sentence makes the answer hallucinated, as in RAGTruth).
- An answer is flagged as hallucinated if that value is below the threshold ({threshold:.2f}).

**Evaluation:** precision, recall, F1 and accuracy are computed against RAGTruth's expert labels
(hallucinated = 1). Precision = of the answers we flagged, how many were truly hallucinated.
Recall = of the truly hallucinated answers, how many we caught.
""")