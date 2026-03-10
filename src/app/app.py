import os
import json
import uuid
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CATALOG = "main"
SCHEMA = "jayd_poc"
MODEL = "databricks-meta-llama-3-3-70b-instruct"
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "d85fb7ed40320552")

st.set_page_config(page_title="Jayd — Prompt Intelligence", layout="wide", page_icon="🧠")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        border: 1px solid rgba(108,99,255,0.2);
    }
    .main-header h1 { color: #e2e8f0; margin: 0; font-size: 2.2rem; }
    .main-header p  { color: #94a3b8; margin: 0.5rem 0 0 0; }
    .score-card {
        background: #1e293b; border-radius: 10px; padding: 1.2rem;
        text-align: center; border: 1px solid #334155;
    }
    .score-card h3 { color: #94a3b8; font-size: 0.85rem; margin: 0;
                     text-transform: uppercase; letter-spacing: 1px; }
    .score-card h1 { margin: 0.3rem 0; font-size: 2rem; }
    .score-green  { color: #22c55e; }
    .score-yellow { color: #eab308; }
    .score-red    { color: #ef4444; }
    .prompt-card {
        background: #1e293b; border-radius: 10px; padding: 1.2rem;
        border: 1px solid #334155; margin-bottom: 1rem;
    }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
             font-size: 0.8rem; font-weight: 600; }
    .badge-green  { background: rgba(34,197,94,0.13); color: #22c55e; }
    .badge-yellow { background: rgba(234,179,8,0.13); color: #eab308; }
    .badge-red    { background: rgba(239,68,68,0.13); color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Database helpers  (databricks-sql-connector — proven pattern)
# ---------------------------------------------------------------------------

def _get_sql_connection():
    """Return a databricks-sql-connector connection using SP / user auth."""
    from databricks import sql as dbsql
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()                       # auto-detects env
    auth = w.config.authenticate()              # dict with Authorization header
    token = auth.get("Authorization", "").replace("Bearer ", "")
    host = w.config.host.replace("https://", "").replace("http://", "")

    return dbsql.connect(
        server_hostname=host,
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        access_token=token,
    )


def run_query(sql: str) -> pd.DataFrame:
    """Execute SELECT and return a DataFrame (empty on error)."""
    try:
        conn = _get_sql_connection()
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame(rows, columns=cols)
    except Exception as exc:
        st.error(f"SQL error: {exc}")
        return pd.DataFrame()


def run_exec(sql: str):
    """Execute INSERT / DELETE (no result set)."""
    conn = _get_sql_connection()
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.close()


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """Call Foundation Model API via simple REST (works with any SDK version)."""
    from databricks.sdk import WorkspaceClient
    import requests as _req

    w = WorkspaceClient()
    auth = w.config.authenticate()
    host = w.config.host.rstrip("/")

    resp = _req.post(
        f"{host}/serving-endpoints/{MODEL}/invocations",
        headers={**auth, "Content-Type": "application/json"},
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.3,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def evaluate_prompt(text: str) -> dict:
    """Score a prompt via LLM — returns parsed dict."""
    instruction = (
        "You are a prompt engineering expert. Evaluate the following prompt on "
        "a scale of 0-100 for each dimension. Return ONLY a valid JSON object "
        "(no markdown, no code fences, no explanation) with these exact keys: "
        "overall_score (int), clarity_score (int), specificity_score (int), "
        "context_score (int), structure_score (int), improvement_suggestion "
        "(string, 1-2 sentences), improved_prompt (string, rewritten better "
        "version).\n\nPrompt to evaluate: \"" + text + "\""
    )
    raw = call_llm(instruction)

    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    # Find first { ... } block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]

    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Data loader (cached 30 s)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_silver():
    df = run_query(
        f"SELECT * FROM {CATALOG}.{SCHEMA}.silver_evaluated_prompts "
        f"ORDER BY evaluated_at DESC"
    )
    for c in ["overall_score", "clarity_score", "specificity_score",
              "context_score", "structure_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Tiny visual helpers
# ---------------------------------------------------------------------------

def _color(v):
    try:
        v = float(v)
    except Exception:
        return "gray"
    return "#22c55e" if v >= 70 else "#eab308" if v >= 40 else "#ef4444"


def _cls(v):
    try:
        v = float(v)
    except Exception:
        return ""
    return "score-green" if v >= 70 else "score-yellow" if v >= 40 else "score-red"


def _badge(v):
    try:
        v = float(v)
    except Exception:
        return ""
    return "badge-green" if v >= 70 else "badge-yellow" if v >= 40 else "badge-red"


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🧠 Jayd — Prompt Intelligence Platform</h1>
    <p>Score, improve, and execute prompts at enterprise scale — powered by Databricks &amp; Llama 3.3</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Prompt Lab", "📊 Analytics", "📚 Template Library", "📡 Live Feed"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1  — Prompt Lab
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Prompt Intelligence Lab")
    st.caption("Enter a prompt → get it scored, improved, and executed against Llama 3.3 70B.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    prompt = st.chat_input("Enter your prompt to evaluate…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Evaluating your prompt…"):
                try:
                    ev = evaluate_prompt(prompt)
                    o = ev.get("overall_score", 0)
                    cl = ev.get("clarity_score", 0)
                    sp = ev.get("specificity_score", 0)
                    cx = ev.get("context_score", 0)
                    sr = ev.get("structure_score", 0)

                    cols = st.columns(5)
                    for c, (lbl, val) in zip(cols, [
                        ("Overall", o), ("Clarity", cl), ("Specificity", sp),
                        ("Context", cx), ("Structure", sr),
                    ]):
                        c.markdown(
                            f'<div class="score-card"><h3>{lbl}</h3>'
                            f'<h1 style="color:{_color(val)}">{val}</h1></div>',
                            unsafe_allow_html=True,
                        )

                    fig = go.Figure(data=go.Scatterpolar(
                        r=[cl, sp, cx, sr, cl],
                        theta=["Clarity", "Specificity", "Context",
                               "Structure", "Clarity"],
                        fill="toself",
                        fillcolor="rgba(108,99,255,0.2)",
                        line=dict(color="#6c63ff", width=2),
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100],
                                            showticklabels=False,
                                            gridcolor="#334155"),
                            angularaxis=dict(gridcolor="#334155",
                                             linecolor="#334155"),
                            bgcolor="#0e1117",
                        ),
                        showlegend=False, paper_bgcolor="#0e1117",
                        font=dict(color="#e2e8f0"),
                        height=300, margin=dict(l=40, r=40, t=20, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    sug = ev.get("improvement_suggestion", "N/A")
                    imp = ev.get("improved_prompt", "N/A")
                    st.markdown("**Improvement Suggestion:**")
                    st.info(sug)
                    st.markdown("**Improved Prompt:**")
                    st.success(imp)

                    st.session_state["last_improved"] = imp

                    # persist to DB (non-blocking)
                    try:
                        pid = f"APP-{uuid.uuid4().hex[:8].upper()}"
                        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        safe = prompt.replace("'", "''")
                        safe_sug = str(sug).replace("'", "''")
                        safe_imp = str(imp).replace("'", "''")
                        run_exec(
                            f"INSERT INTO {CATALOG}.{SCHEMA}.bronze_prompts "
                            f"VALUES('{pid}','{safe}','app_user','App',"
                            f"'user_input','{now}','app','{now}')"
                        )
                        run_exec(
                            f"INSERT INTO {CATALOG}.{SCHEMA}.silver_evaluated_prompts "
                            f"VALUES('{pid}','{safe}','app_user','App',"
                            f"'user_input','{now}','app',{o},{cl},{sp},{cx},{sr},"
                            f"'{safe_sug}','{safe_imp}','{now}')"
                        )
                    except Exception:
                        pass

                    md = (f"**Score: {o}/100** | Clarity {cl} | Specificity {sp}"
                          f" | Context {cx} | Structure {sr}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": md})

                except Exception as exc:
                    st.error(f"Evaluation error: {exc}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {exc}"})

    if st.session_state.get("last_improved"):
        if st.button("Execute Improved Prompt", type="primary"):
            with st.chat_message("assistant"):
                with st.spinner("Generating response…"):
                    try:
                        resp = call_llm(st.session_state["last_improved"])
                        st.markdown("**LLM Response:**")
                        st.markdown(resp)
                        st.session_state.messages.append(
                            {"role": "assistant",
                             "content": f"**Response:**\n\n{resp}"})
                    except Exception as exc:
                        st.error(f"Execution error: {exc}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2  — Analytics
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Prompt Quality Analytics")
    df = load_silver()

    if df.empty:
        st.warning("No evaluated prompts yet — run the pipeline first.")
    else:
        total = len(df)
        avg = df["overall_score"].mean()
        pct70 = (df["overall_score"] >= 70).sum() / total * 100
        top_d = df.groupby("department")["overall_score"].mean().idxmax()

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="score-card"><h3>Total Prompts</h3>'
                    f'<h1 style="color:#6c63ff">{total}</h1></div>',
                    unsafe_allow_html=True)
        k2.markdown(f'<div class="score-card"><h3>Avg Score</h3>'
                    f'<h1 class="{_cls(avg)}">{avg:.0f}</h1></div>',
                    unsafe_allow_html=True)
        k3.markdown(f'<div class="score-card"><h3>% Above 70</h3>'
                    f'<h1 class="{_cls(pct70)}">{pct70:.0f}%</h1></div>',
                    unsafe_allow_html=True)
        k4.markdown(f'<div class="score-card"><h3>Top Dept</h3>'
                    f'<h1 style="color:#6c63ff;font-size:1.3rem">{top_d}</h1>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Score Distribution")
            fig = px.histogram(df, x="overall_score", nbins=20,
                               color_discrete_sequence=["#6c63ff"])
            fig.update_layout(paper_bgcolor="#0e1117",
                              plot_bgcolor="#0e1117",
                              font=dict(color="#e2e8f0"),
                              xaxis=dict(gridcolor="#334155"),
                              yaxis=dict(gridcolor="#334155"),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("##### Avg Score by Category")
            ca = (df.groupby("category")["overall_score"].mean()
                    .reset_index().sort_values("overall_score"))
            fig = px.bar(ca, x="overall_score", y="category",
                         orientation="h", color="overall_score",
                         color_continuous_scale=["#ef4444","#eab308","#22c55e"])
            fig.update_layout(paper_bgcolor="#0e1117",
                              plot_bgcolor="#0e1117",
                              font=dict(color="#e2e8f0"),
                              xaxis=dict(gridcolor="#334155"),
                              yaxis=dict(gridcolor="#334155"),
                              showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Score Dimensions by Department")
        dims = ["clarity_score","specificity_score","context_score","structure_score"]
        ds = df.groupby("department")[dims].mean()
        colors = ["#6c63ff","#22c55e","#eab308","#ef4444","#3b82f6","#ec4899"]
        fig = go.Figure()
        for i, (dept, row) in enumerate(ds.iterrows()):
            vals = [row[d] for d in dims] + [row[dims[0]]]
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=["Clarity","Specificity","Context","Structure","Clarity"],
                fill="toself", name=dept,
                line=dict(color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)] + "22",
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100],
                                       gridcolor="#334155"),
                       angularaxis=dict(gridcolor="#334155"),
                       bgcolor="#0e1117"),
            paper_bgcolor="#0e1117", font=dict(color="#e2e8f0"), height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Bottom 10 — Improvement Opportunities")
        bot = df.nsmallest(10, "overall_score")[
            ["prompt_id","prompt_text","department","category",
             "overall_score","improvement_suggestion"]]
        st.dataframe(bot, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3  — Template Library
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Template Library")
    st.caption("Top-scoring improved prompts you can reuse.")
    df = load_silver()
    if df.empty:
        st.warning("No data yet.")
    else:
        top = df[df["overall_score"] >= 70].nlargest(20, "overall_score")
        if top.empty:
            st.info("No prompts scored 70+ yet.")
        else:
            cats = ["All"] + sorted(top["category"].dropna().unique().tolist())
            sel = st.selectbox("Filter by category", cats, key="tpl_cat")
            if sel != "All":
                top = top[top["category"] == sel]
            for _, r in top.iterrows():
                s = r["overall_score"]
                st.markdown(
                    f'<div class="prompt-card">'
                    f'<span style="color:#94a3b8;font-size:.85rem">'
                    f'{r["category"]} &middot; {r["department"]}</span>'
                    f'&nbsp;<span class="badge {_badge(s)}">{int(s)}/100</span>'
                    f'</div>', unsafe_allow_html=True)
                a, b = st.columns(2)
                a.text_area("Original", r["prompt_text"], height=90,
                            key=f"o_{r['prompt_id']}", disabled=True)
                b.text_area("Improved", str(r.get("improved_prompt","N/A")),
                            height=90, key=f"i_{r['prompt_id']}", disabled=True)
                if st.button("Use Template", key=f"u_{r['prompt_id']}"):
                    st.session_state["template_prompt"] = r.get(
                        "improved_prompt", r["prompt_text"])
                    st.toast("Copied! Go to Prompt Lab.")
                st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4  — Live Feed
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Live Prompt Feed")
    df = load_silver()
    if df.empty:
        st.warning("No data yet.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            srcs = ["All"] + sorted(df["source"].dropna().unique().tolist())
            ss = st.selectbox("Source", srcs)
        with f2:
            cats = ["All"] + sorted(df["category"].dropna().unique().tolist())
            sc = st.selectbox("Category", cats, key="fc")
        with f3:
            deps = ["All"] + sorted(df["department"].dropna().unique().tolist())
            sd = st.selectbox("Department", deps)

        fd = df.copy()
        if ss != "All": fd = fd[fd["source"] == ss]
        if sc != "All": fd = fd[fd["category"] == sc]
        if sd != "All": fd = fd[fd["department"] == sd]

        st.markdown(f"**Showing {len(fd)} prompts**")

        def _cs(v):
            try:
                v = float(v)
            except Exception:
                return ""
            if v >= 70: return "background-color:rgba(34,197,94,.2);color:#22c55e"
            if v >= 40: return "background-color:rgba(234,179,8,.2);color:#eab308"
            return "background-color:rgba(239,68,68,.2);color:#ef4444"

        show = ["prompt_id","prompt_text","department","category","source",
                "overall_score","clarity_score","specificity_score",
                "context_score","structure_score"]
        show = [c for c in show if c in fd.columns]
        styled = fd[show].style.map(_cs, subset=[
            c for c in ["overall_score","clarity_score","specificity_score",
                        "context_score","structure_score"] if c in fd.columns])
        st.dataframe(styled, use_container_width=True, hide_index=True,
                     height=500)

        st.markdown("##### Prompt Details")
        if not fd.empty:
            sel_id = st.selectbox("Select prompt", fd["prompt_id"].tolist())
            if sel_id:
                r = fd[fd["prompt_id"] == sel_id].iloc[0]
                with st.expander(f"Details — {sel_id}", expanded=True):
                    st.markdown(f"**Prompt:** {r['prompt_text']}")
                    st.markdown(
                        f"**Overall:** {r['overall_score']} · "
                        f"Clarity {r['clarity_score']} · "
                        f"Specificity {r['specificity_score']} · "
                        f"Context {r['context_score']} · "
                        f"Structure {r['structure_score']}")
                    st.markdown(f"**Suggestion:** {r.get('improvement_suggestion','')}")
                    st.markdown(f"**Improved:** {r.get('improved_prompt','')}")
