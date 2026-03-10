import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import json
import uuid
from datetime import datetime

# --- Config ---
st.set_page_config(page_title="Jayd — Prompt Intelligence", layout="wide", page_icon="🧠")

CATALOG = "main"
SCHEMA = "jayd_poc"
MODEL = "databricks-meta-llama-3-3-70b-instruct"

# --- Styling ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        border: 1px solid #6c63ff33;
    }
    .main-header h1 { color: #e2e8f0; margin: 0; font-size: 2.2rem; }
    .main-header p { color: #94a3b8; margin: 0.5rem 0 0 0; }
    .score-card {
        background: #1e293b; border-radius: 10px; padding: 1.2rem;
        text-align: center; border: 1px solid #334155;
    }
    .score-card h3 { color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; letter-spacing: 1px; }
    .score-card h1 { margin: 0.3rem 0; font-size: 2rem; }
    .score-green { color: #22c55e; }
    .score-yellow { color: #eab308; }
    .score-red { color: #ef4444; }
    .prompt-card {
        background: #1e293b; border-radius: 10px; padding: 1.2rem;
        border: 1px solid #334155; margin-bottom: 1rem;
    }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600;
    }
    .badge-green { background: #22c55e22; color: #22c55e; }
    .badge-yellow { background: #eab30822; color: #eab308; }
    .badge-red { background: #ef444422; color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# --- Init ---
@st.cache_resource
def get_workspace_client():
    return WorkspaceClient()


def get_connection():
    """Get a SQL connection using databricks-sql-connector."""
    from databricks.sdk import WorkspaceClient
    import databricks.sql

    w = get_workspace_client()
    host = w.config.host.replace("https://", "")
    token = w.config.authenticate()

    import os
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
    if not warehouse_id:
        # Try to find a running warehouse
        warehouses = list(w.warehouses.list())
        for wh in warehouses:
            if wh.state and wh.state.value == "RUNNING":
                warehouse_id = wh.id
                break
        if not warehouse_id and warehouses:
            warehouse_id = warehouses[0].id

    return databricks.sql.connect(
        server_hostname=host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: {"Authorization": f"Bearer {token}"}
    )


@st.cache_data(ttl=30)
def load_silver_data():
    """Load evaluated prompts from silver table."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {CATALOG}.{SCHEMA}.silver_evaluated_prompts ORDER BY evaluated_at DESC")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        df = pd.DataFrame(rows, columns=columns)
        return df
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return pd.DataFrame()


def score_color(score):
    if score is None:
        return "gray"
    if score >= 70:
        return "#22c55e"
    elif score >= 40:
        return "#eab308"
    return "#ef4444"


def score_class(score):
    if score is None:
        return ""
    if score >= 70:
        return "score-green"
    elif score >= 40:
        return "score-yellow"
    return "score-red"


def score_badge(score):
    if score is None:
        return ""
    if score >= 70:
        return "badge-green"
    elif score >= 40:
        return "badge-yellow"
    return "badge-red"


def call_llm(prompt: str) -> str:
    """Call the LLM via workspace client."""
    w = get_workspace_client()
    response = w.serving_endpoints.query(
        name=MODEL,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=2048,
        temperature=0.3,
    )
    return response.choices[0].message.content


def evaluate_prompt(prompt_text: str) -> dict:
    """Score a prompt using the LLM."""
    system_prompt = (
        "You are a prompt engineering expert. Evaluate the following prompt on a scale of 0-100 for each dimension. "
        "Return ONLY valid JSON (no markdown, no explanation, no code fences) with these exact keys: "
        "overall_score (int), clarity_score (int), specificity_score (int), context_score (int), structure_score (int), "
        "improvement_suggestion (string, 1-2 sentences), improved_prompt (string, the rewritten better version). "
        f'Prompt to evaluate: "{prompt_text}"'
    )
    raw = call_llm(system_prompt)
    # Clean potential markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def insert_prompt_and_evaluation(prompt_text, evaluation):
    """Insert a user-submitted prompt into both tables."""
    prompt_id = f"APP-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    # Insert into bronze
    cursor.execute(f"""
        INSERT INTO {CATALOG}.{SCHEMA}.bronze_prompts
        (prompt_id, prompt_text, user_id, department, category, submitted_at, source, _ingested_at)
        VALUES ('{prompt_id}', '{prompt_text.replace("'", "''")}', 'app_user', 'App', 'user_input',
                '{now}', 'app', '{now}')
    """)

    # Insert into silver
    cursor.execute(f"""
        INSERT INTO {CATALOG}.{SCHEMA}.silver_evaluated_prompts
        (prompt_id, prompt_text, user_id, department, category, submitted_at, source,
         overall_score, clarity_score, specificity_score, context_score, structure_score,
         improvement_suggestion, improved_prompt, evaluated_at)
        VALUES ('{prompt_id}', '{prompt_text.replace("'", "''")}', 'app_user', 'App', 'user_input',
                '{now}', 'app',
                {evaluation.get('overall_score', 0)}, {evaluation.get('clarity_score', 0)},
                {evaluation.get('specificity_score', 0)}, {evaluation.get('context_score', 0)},
                {evaluation.get('structure_score', 0)},
                '{str(evaluation.get("improvement_suggestion", "")).replace("'", "''")}',
                '{str(evaluation.get("improved_prompt", "")).replace("'", "''")}',
                '{now}')
    """)

    cursor.close()
    conn.close()


# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🧠 Jayd — Prompt Intelligence Platform</h1>
    <p>Score, improve, and execute prompts at enterprise scale — powered by Databricks & Llama 3.3</p>
</div>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["💬 Prompt Lab", "📊 Analytics", "📚 Template Library", "📡 Live Feed"])

# ============================================================
# TAB 1: Prompt Lab
# ============================================================
with tab1:
    st.subheader("Prompt Intelligence Lab")
    st.caption("Enter a prompt to get it scored, improved, and executed against Llama 3.3 70B.")

    # Session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "template_prompt" not in st.session_state:
        st.session_state.template_prompt = None

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Check for template redirect
    default_prompt = ""
    if st.session_state.template_prompt:
        default_prompt = st.session_state.template_prompt
        st.session_state.template_prompt = None

    prompt = st.chat_input("Enter your prompt to evaluate...", key="prompt_input")

    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Evaluating your prompt..."):
                try:
                    evaluation = evaluate_prompt(prompt)

                    overall = evaluation.get("overall_score", 0)
                    clarity = evaluation.get("clarity_score", 0)
                    specificity = evaluation.get("specificity_score", 0)
                    context = evaluation.get("context_score", 0)
                    structure = evaluation.get("structure_score", 0)

                    # Score display
                    cols = st.columns(5)
                    for c, (label, val) in zip(cols, [
                        ("Overall", overall), ("Clarity", clarity),
                        ("Specificity", specificity), ("Context", context), ("Structure", structure)
                    ]):
                        color = score_color(val)
                        c.markdown(f"""
                        <div class="score-card">
                            <h3>{label}</h3>
                            <h1 style="color: {color}">{val}</h1>
                        </div>
                        """, unsafe_allow_html=True)

                    # Radar chart
                    fig = go.Figure(data=go.Scatterpolar(
                        r=[clarity, specificity, context, structure, clarity],
                        theta=["Clarity", "Specificity", "Context", "Structure", "Clarity"],
                        fill="toself",
                        fillcolor="rgba(108, 99, 255, 0.2)",
                        line=dict(color="#6c63ff", width=2),
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#334155"),
                            angularaxis=dict(gridcolor="#334155", linecolor="#334155"),
                            bgcolor="#0e1117",
                        ),
                        showlegend=False,
                        paper_bgcolor="#0e1117",
                        font=dict(color="#e2e8f0"),
                        height=300,
                        margin=dict(l=40, r=40, t=20, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Improvement suggestion
                    suggestion = evaluation.get("improvement_suggestion", "N/A")
                    improved = evaluation.get("improved_prompt", "N/A")

                    st.markdown("**💡 Improvement Suggestion:**")
                    st.info(suggestion)

                    st.markdown("**✨ Improved Prompt:**")
                    st.success(improved)

                    # Save to DB
                    try:
                        insert_prompt_and_evaluation(prompt, evaluation)
                    except Exception:
                        pass  # Non-blocking

                    # Store for execute button
                    st.session_state["last_improved_prompt"] = improved

                    response_md = f"**Score: {overall}/100** | Clarity: {clarity} | Specificity: {specificity} | Context: {context} | Structure: {structure}"
                    st.session_state.messages.append({"role": "assistant", "content": response_md})

                except Exception as e:
                    st.error(f"Evaluation error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

    # Execute improved prompt
    if st.session_state.get("last_improved_prompt"):
        if st.button("🚀 Execute Improved Prompt", type="primary"):
            improved = st.session_state["last_improved_prompt"]
            with st.chat_message("assistant"):
                with st.spinner("🤖 Generating response with improved prompt..."):
                    try:
                        response = call_llm(improved)
                        st.markdown("**📝 LLM Response:**")
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": f"**Response to improved prompt:**\n\n{response}"})
                    except Exception as e:
                        st.error(f"Execution error: {e}")

# ============================================================
# TAB 2: Analytics
# ============================================================
with tab2:
    st.subheader("Prompt Quality Analytics")
    df = load_silver_data()

    if df.empty:
        st.warning("No evaluated prompts found. Run the pipeline first.")
    else:
        # Ensure numeric columns
        for col_name in ["overall_score", "clarity_score", "specificity_score", "context_score", "structure_score"]:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        total = len(df)
        avg_score = df["overall_score"].mean()
        pct_above_70 = (df["overall_score"] >= 70).sum() / total * 100 if total > 0 else 0
        top_dept = df.groupby("department")["overall_score"].mean().idxmax() if not df.empty else "N/A"

        k1.markdown(f'<div class="score-card"><h3>Total Prompts</h3><h1 style="color:#6c63ff">{total}</h1></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="score-card"><h3>Avg Score</h3><h1 class="{score_class(avg_score)}">{avg_score:.0f}</h1></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="score-card"><h3>% Above 70</h3><h1 class="{score_class(pct_above_70)}">{pct_above_70:.0f}%</h1></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="score-card"><h3>Top Department</h3><h1 style="color:#6c63ff;font-size:1.3rem">{top_dept}</h1></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Charts
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("##### Score Distribution")
            fig_hist = px.histogram(
                df, x="overall_score", nbins=20,
                color_discrete_sequence=["#6c63ff"],
                labels={"overall_score": "Overall Score"},
            )
            fig_hist.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155"), showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.markdown("##### Average Score by Category")
            cat_avg = df.groupby("category")["overall_score"].mean().reset_index().sort_values("overall_score", ascending=True)
            fig_cat = px.bar(
                cat_avg, x="overall_score", y="category", orientation="h",
                color="overall_score", color_continuous_scale=["#ef4444", "#eab308", "#22c55e"],
                labels={"overall_score": "Avg Score", "category": "Category"},
            )
            fig_cat.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155"), showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        # Radar by department
        st.markdown("##### Score Dimensions by Department")
        dept_scores = df.groupby("department")[["clarity_score", "specificity_score", "context_score", "structure_score"]].mean()
        fig_radar = go.Figure()
        colors = ["#6c63ff", "#22c55e", "#eab308", "#ef4444", "#3b82f6", "#ec4899"]
        for i, (dept, row) in enumerate(dept_scores.iterrows()):
            vals = [row["clarity_score"], row["specificity_score"], row["context_score"], row["structure_score"], row["clarity_score"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=["Clarity", "Specificity", "Context", "Structure", "Clarity"],
                fill="toself", name=dept, line=dict(color=colors[i % len(colors)]),
                fillcolor=f"{colors[i % len(colors)]}22",
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155"), angularaxis=dict(gridcolor="#334155"), bgcolor="#0e1117"),
            paper_bgcolor="#0e1117", font=dict(color="#e2e8f0"), height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Bottom 10
        st.markdown("##### ⚠️ Bottom 10 Prompts (Improvement Opportunities)")
        bottom = df.nsmallest(10, "overall_score")[["prompt_id", "prompt_text", "department", "category", "overall_score", "improvement_suggestion"]]
        st.dataframe(bottom, use_container_width=True, hide_index=True)

# ============================================================
# TAB 3: Template Library
# ============================================================
with tab3:
    st.subheader("Template Library")
    st.caption("Top-scoring improved prompts you can reuse.")

    df = load_silver_data()
    if df.empty:
        st.warning("No evaluated prompts found.")
    else:
        for col_name in ["overall_score"]:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        top = df[df["overall_score"] >= 70].nlargest(20, "overall_score")

        if top.empty:
            st.info("No prompts scored above 70 yet.")
        else:
            # Filters
            categories = ["All"] + sorted(top["category"].dropna().unique().tolist())
            selected_cat = st.selectbox("Filter by category", categories, key="template_cat")
            if selected_cat != "All":
                top = top[top["category"] == selected_cat]

            for _, row in top.iterrows():
                score = row["overall_score"]
                badge = score_badge(score)
                with st.container():
                    st.markdown(f"""
                    <div class="prompt-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                            <span style="color:#94a3b8; font-size:0.85rem;">{row['category']} · {row['department']}</span>
                            <span class="badge {badge}">{score}/100</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Original:**")
                        st.text_area("", row["prompt_text"], height=100, key=f"orig_{row['prompt_id']}", disabled=True)
                    with col_b:
                        st.markdown("**Improved:**")
                        st.text_area("", row.get("improved_prompt", "N/A"), height=100, key=f"imp_{row['prompt_id']}", disabled=True)

                    if st.button(f"Use Template", key=f"use_{row['prompt_id']}"):
                        st.session_state.template_prompt = row.get("improved_prompt", row["prompt_text"])
                        st.toast("Template copied! Switch to Prompt Lab tab to use it.")

                    st.markdown("---")

# ============================================================
# TAB 4: Live Feed
# ============================================================
with tab4:
    st.subheader("Live Prompt Feed")

    df = load_silver_data()
    if df.empty:
        st.warning("No evaluated prompts found.")
    else:
        for col_name in ["overall_score", "clarity_score", "specificity_score", "context_score", "structure_score"]:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            sources = ["All"] + sorted(df["source"].dropna().unique().tolist())
            sel_source = st.selectbox("Source", sources)
        with f2:
            categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
            sel_cat = st.selectbox("Category", categories, key="feed_cat")
        with f3:
            departments = ["All"] + sorted(df["department"].dropna().unique().tolist())
            sel_dept = st.selectbox("Department", departments)

        filtered = df.copy()
        if sel_source != "All":
            filtered = filtered[filtered["source"] == sel_source]
        if sel_cat != "All":
            filtered = filtered[filtered["category"] == sel_cat]
        if sel_dept != "All":
            filtered = filtered[filtered["department"] == sel_dept]

        st.markdown(f"**Showing {len(filtered)} prompts**")

        # Color-coded dataframe
        def color_score(val):
            try:
                v = float(val)
                if v >= 70:
                    return "background-color: #22c55e33; color: #22c55e"
                elif v >= 40:
                    return "background-color: #eab30833; color: #eab308"
                return "background-color: #ef444433; color: #ef4444"
            except (ValueError, TypeError):
                return ""

        display_cols = ["prompt_id", "prompt_text", "department", "category", "source",
                        "overall_score", "clarity_score", "specificity_score", "context_score", "structure_score"]
        display_df = filtered[display_cols].copy()

        styled = display_df.style.map(
            color_score,
            subset=["overall_score", "clarity_score", "specificity_score", "context_score", "structure_score"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

        # Expandable details
        st.markdown("##### Prompt Details")
        selected_id = st.selectbox("Select a prompt to view details", filtered["prompt_id"].tolist())
        if selected_id:
            row = filtered[filtered["prompt_id"] == selected_id].iloc[0]
            with st.expander(f"Details for {selected_id}", expanded=True):
                st.markdown(f"**Original Prompt:** {row['prompt_text']}")
                st.markdown(f"**Overall Score:** {row['overall_score']}")
                st.markdown(f"**Clarity:** {row['clarity_score']} | **Specificity:** {row['specificity_score']} | **Context:** {row['context_score']} | **Structure:** {row['structure_score']}")
                st.markdown(f"**Suggestion:** {row.get('improvement_suggestion', 'N/A')}")
                st.markdown(f"**Improved Prompt:** {row.get('improved_prompt', 'N/A')}")
