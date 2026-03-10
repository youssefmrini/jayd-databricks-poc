import os
import json
import uuid
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
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "d85fb7ed40320552")

AVAILABLE_MODELS = [
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-meta-llama-3-1-405b-instruct",
    "databricks-dbrx-instruct",
    "databricks-mixtral-8x7b-instruct",
]

# ---------------------------------------------------------------------------
# 15 curated prompt templates by Business Unit
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = [
    {
        "bu": "Marketing",
        "title": "Campaign Performance Deep-Dive",
        "prompt": (
            "Analyze the Q1 2026 digital marketing campaign for our B2B SaaS product. "
            "The campaign ran across LinkedIn Ads, Google Search, and email nurture sequences "
            "targeting VP-level decision-makers in financial services. Total spend was $45,000. "
            "Results: 12,400 impressions, 890 clicks, 67 MQLs, 23 SQLs, and 4 closed-won deals "
            "worth $320K ARR. Calculate CAC, ROAS, and conversion rates at each funnel stage. "
            "Compare against industry benchmarks for B2B SaaS. Identify the highest-performing "
            "channel and recommend budget reallocation for Q2."
        ),
    },
    {
        "bu": "Sales",
        "title": "Competitive Deal Strategy",
        "prompt": (
            "We are competing against Snowflake and Databricks in a $2M enterprise deal with a "
            "Fortune 500 retail company. The customer currently uses Hadoop on-prem with Hive and "
            "Spark. Their key requirements are: real-time inventory analytics, ML-based demand "
            "forecasting, and a unified governance layer across 3 cloud regions. Our platform "
            "strengths are lakehouse architecture and built-in ML. Draft a competitive positioning "
            "document with: (1) a SWOT analysis for each vendor, (2) three killer differentiators "
            "to highlight in our next executive briefing, and (3) potential objections the customer "
            "might raise with suggested rebuttals."
        ),
    },
    {
        "bu": "Engineering",
        "title": "Code Review & Refactoring Plan",
        "prompt": (
            "Review the following Python data pipeline that processes 50M rows daily from Kafka "
            "into Delta Lake. The pipeline currently takes 4.2 hours to complete and occasionally "
            "fails with OOM errors on the executor nodes. Identify: (1) performance bottlenecks "
            "such as unnecessary shuffles, suboptimal partitioning, or missing Z-ordering, "
            "(2) reliability gaps including missing retry logic, dead-letter queues, or schema "
            "evolution handling, (3) code quality issues like hardcoded values, missing type hints, "
            "or absent error handling. Provide a prioritized refactoring plan with estimated effort "
            "for each improvement."
        ),
    },
    {
        "bu": "Human Resources",
        "title": "Job Description Generator",
        "prompt": (
            "Create a compelling job description for a Senior Machine Learning Engineer role at "
            "a Series C fintech startup (450 employees, $120M funding). The role sits in the "
            "Risk & Fraud Detection team and requires 5+ years of experience with production ML "
            "systems. Tech stack: Python, PySpark, MLflow, Databricks, and AWS. The ideal candidate "
            "has experience with real-time feature stores, model monitoring, and regulatory compliance "
            "(SOC2, PCI-DSS). Include: role summary, key responsibilities (6-8 bullets), required "
            "qualifications, preferred qualifications, compensation range ($180K-$240K + equity), "
            "and a section on company culture emphasizing remote-first, learning budget, and diversity "
            "commitment. Tone should be professional yet approachable."
        ),
    },
    {
        "bu": "Finance",
        "title": "Financial Forecasting Analysis",
        "prompt": (
            "Build a 12-month rolling revenue forecast for our SaaS business using the following "
            "inputs: current ARR of $18.5M, net revenue retention of 112%, gross churn of 8% "
            "annually, average new ACV of $45K, current pipeline of $6.2M (weighted), sales cycle "
            "of 90 days, and seasonal patterns showing Q4 accounting for 35% of new bookings. "
            "Model three scenarios: (1) base case with current growth trajectory, (2) optimistic "
            "with planned product launch boosting win rates by 15%, (3) conservative with economic "
            "downturn extending sales cycles by 30%. Present the output as a month-by-month table "
            "with MRR, new ARR, churned ARR, and net ARR for each scenario."
        ),
    },
    {
        "bu": "Legal",
        "title": "Contract Clause Review",
        "prompt": (
            "Review the following enterprise SaaS Master Service Agreement for a healthcare customer "
            "subject to HIPAA regulations. Flag any clauses that: (1) create unlimited liability "
            "exposure for the vendor, (2) contain data residency requirements that conflict with "
            "our multi-region cloud architecture, (3) impose SLA penalties exceeding industry "
            "standard (99.9% uptime), (4) include non-standard IP assignment or work-for-hire "
            "provisions, (5) require BAA terms that go beyond standard HIPAA requirements. For each "
            "flagged clause, provide: the original text, the risk level (high/medium/low), a plain "
            "English explanation of the issue, and suggested counter-language."
        ),
    },
    {
        "bu": "Product",
        "title": "Feature Prioritization Framework",
        "prompt": (
            "We have 12 feature requests in our product backlog for Q2 2026. Using the RICE "
            "scoring framework (Reach, Impact, Confidence, Effort), prioritize these features: "
            "(1) AI-powered auto-complete for SQL editor, (2) native Git integration for notebooks, "
            "(3) role-based access control for dashboards, (4) real-time collaboration on queries, "
            "(5) automated data quality monitoring, (6) Slack integration for alerts, "
            "(7) custom visualization builder, (8) API rate limiting dashboard, "
            "(9) multi-language support (FR, DE, ES), (10) SSO with Okta/Azure AD, "
            "(11) data catalog search improvements, (12) mobile-responsive dashboard viewer. "
            "Our ICP is mid-market data teams (50-500 employees). Current NPS is 42. "
            "Top churn reason is lack of governance features. Output a ranked table with RICE "
            "scores and a recommended Q2 roadmap."
        ),
    },
    {
        "bu": "Data & Analytics",
        "title": "Data Quality Assessment",
        "prompt": (
            "Perform a comprehensive data quality assessment on our customer_360 table that feeds "
            "the executive KPI dashboard. The table has 2.3M rows across 47 columns, sourced from "
            "Salesforce CRM, Stripe billing, Mixpanel product analytics, and Zendesk support. "
            "Check for: (1) completeness — percentage of NULLs per column, flag any >5%, "
            "(2) consistency — conflicting values across source systems (e.g., different customer "
            "names in CRM vs billing), (3) timeliness — records with stale data (>7 days since "
            "source update), (4) uniqueness — duplicate customer records based on email and domain, "
            "(5) accuracy — revenue figures that deviate >20% from Stripe source of truth. "
            "Output a data quality scorecard with red/amber/green ratings and remediation steps."
        ),
    },
    {
        "bu": "Customer Support",
        "title": "Escalation Response Template",
        "prompt": (
            "A P1 enterprise customer (ARR $450K, 3-year contract renewing in 60 days) has "
            "escalated a critical issue: their production ETL pipeline on our platform has been "
            "failing intermittently for 72 hours, causing downstream BI dashboards to show stale "
            "data. The customer's VP of Data has emailed our CEO directly. Root cause identified: "
            "a memory leak in our latest platform update (v3.2.1) affecting clusters with >256GB RAM. "
            "Draft: (1) an immediate acknowledgment email to the VP (empathetic, takes ownership, "
            "provides timeline), (2) a technical remediation plan with ETA, (3) a goodwill gesture "
            "proposal (credit, extended support, executive review), and (4) internal post-mortem "
            "action items to prevent recurrence."
        ),
    },
    {
        "bu": "Operations",
        "title": "Process Optimization Audit",
        "prompt": (
            "Audit our order-to-cash (O2C) process for a manufacturing company with $200M annual "
            "revenue. Current process: Sales creates quote in Salesforce (avg 2 days) -> Legal "
            "reviews contract (avg 5 days) -> Finance approves credit (avg 3 days) -> Operations "
            "provisions resources (avg 4 days) -> Customer success onboards (avg 10 days). "
            "Total cycle time: 24 days. Industry benchmark: 12 days. Identify: (1) bottlenecks "
            "and their root causes, (2) steps that can be parallelized, (3) automation opportunities "
            "using existing tools (Salesforce, DocuSign, NetSuite), (4) quick wins achievable in "
            "30 days vs strategic improvements for 90 days. Target: reduce cycle time to 14 days."
        ),
    },
    {
        "bu": "Design",
        "title": "UX Audit & Recommendations",
        "prompt": (
            "Conduct a UX audit of our B2B analytics dashboard based on these findings from "
            "20 user interviews: (1) 65% of users can't find the export button within 30 seconds, "
            "(2) the filter panel takes 8 clicks to apply date range + department + metric type, "
            "(3) chart tooltips truncate after 50 characters, hiding critical data labels, "
            "(4) the color palette fails WCAG AA contrast on the light theme, (5) mobile usage "
            "is 22% but the layout breaks below 768px, (6) average task completion time for "
            "'compare two metrics over time' is 4.2 minutes (target: 1 minute). "
            "For each finding, provide: severity (critical/major/minor), affected user personas, "
            "a specific design recommendation with a wireframe description, and expected impact "
            "on task completion time."
        ),
    },
    {
        "bu": "Research & Development",
        "title": "Research Synthesis Report",
        "prompt": (
            "Synthesize the latest research (2024-2026) on Retrieval-Augmented Generation (RAG) "
            "for enterprise knowledge management. Cover: (1) comparison of chunking strategies "
            "(fixed-size, semantic, recursive) and their impact on retrieval accuracy, "
            "(2) vector database benchmarks (Pinecone vs Weaviate vs ChromaDB vs Databricks "
            "Vector Search) for datasets >10M documents, (3) hybrid search approaches combining "
            "dense and sparse retrieval, (4) evaluation frameworks (RAGAS, DeepEval) and their "
            "limitations, (5) production challenges including hallucination rates, latency at "
            "scale, and cost optimization. Output a structured literature review with key findings, "
            "a comparison matrix, and recommended architecture for a 50K-document enterprise "
            "knowledge base."
        ),
    },
    {
        "bu": "IT & Infrastructure",
        "title": "Incident Response Runbook",
        "prompt": (
            "Create a detailed incident response runbook for a Severity 1 database outage "
            "affecting our production Databricks workspace. The workspace serves 200+ data "
            "engineers and powers 45 customer-facing dashboards. Include: (1) initial triage "
            "checklist (first 15 minutes): who to page, what to check, communication templates, "
            "(2) diagnostic steps: cluster health, driver logs, warehouse query queue, Unity "
            "Catalog metastore status, network connectivity, (3) common root causes and fixes: "
            "credential expiry, quota exceeded, storage mount failures, IAM policy changes, "
            "(4) escalation matrix with response times for each severity level, "
            "(5) post-incident review template. Format as a step-by-step playbook that an "
            "on-call L2 engineer can follow at 3 AM."
        ),
    },
    {
        "bu": "Executive Leadership",
        "title": "Board Presentation Briefing",
        "prompt": (
            "Prepare an executive briefing for our Q1 2026 board meeting. Company context: "
            "Series B SaaS ($32M ARR, 180 employees, 340 customers). Key metrics to present: "
            "ARR growth 45% YoY (target was 50%), NDR 118%, gross margin 72%, burn multiple 1.8x, "
            "CAC payback 14 months, logo churn 12% (up from 9%). New enterprise wins: 3 Fortune "
            "500 logos. Product: launched AI assistant (adopted by 28% of users in 6 weeks). "
            "Team: hired VP Engineering and VP Marketing. Challenges: longer sales cycles in "
            "mid-market, rising cloud infrastructure costs (+18% QoQ). Structure the briefing as: "
            "(1) 3-sentence TL;DR, (2) financial performance vs plan, (3) strategic wins, "
            "(4) areas of concern with mitigation plans, (5) Q2 priorities and board asks."
        ),
    },
    {
        "bu": "Compliance & Risk",
        "title": "Regulatory Impact Assessment",
        "prompt": (
            "Assess the impact of the EU AI Act (effective August 2025) on our ML-powered "
            "credit scoring product used by 12 European banking customers. Our system uses: "
            "(1) a gradient boosting model trained on 5 years of transaction data, "
            "(2) real-time feature engineering from 200+ variables including demographic data, "
            "(3) automated decision-making for loan amounts under EUR 50K. Evaluate: "
            "(a) which risk category our system falls under (likely high-risk per Annex III), "
            "(b) specific compliance requirements: transparency obligations, human oversight "
            "mechanisms, data governance standards, bias testing mandates, (c) gap analysis "
            "between our current practices and required standards, (d) estimated compliance "
            "cost and timeline, (e) recommended action plan with milestones. Reference specific "
            "articles of the EU AI Act where applicable."
        ),
    },
]

st.set_page_config(page_title="Jayd — Prompt Intelligence", layout="wide", page_icon="@")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        border: 1px solid rgba(108,99,255,0.3);
        box-shadow: 0 8px 32px rgba(108,99,255,0.15);
    }
    .main-header h1 { color: #f1f5f9; margin: 0; font-size: 2rem;
                       font-weight: 700; letter-spacing: -0.5px; }
    .main-header p  { color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1rem; }

    .kpi-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-radius: 12px; padding: 1.2rem;
        text-align: center; border: 1px solid #334155;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px);
                      box-shadow: 0 4px 20px rgba(108,99,255,0.2); }
    .kpi-card h3 { color: #94a3b8; font-size: 0.75rem; margin: 0;
                   text-transform: uppercase; letter-spacing: 1.5px; font-weight: 500; }
    .kpi-card h1 { margin: 0.3rem 0 0 0; font-size: 2rem; font-weight: 700; }

    .score-green  { color: #22c55e; }
    .score-yellow { color: #eab308; }
    .score-red    { color: #ef4444; }

    .template-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-radius: 12px; padding: 1.5rem;
        border: 1px solid #334155; margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .template-card:hover { transform: translateY(-2px);
                           box-shadow: 0 4px 20px rgba(108,99,255,0.2); }
    .template-card .bu-tag {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
        background: rgba(108,99,255,0.15); color: #a78bfa;
        margin-bottom: 0.5rem;
    }
    .template-card h4 { color: #f1f5f9; margin: 0.3rem 0; font-size: 1.05rem; }
    .template-card p  { color: #94a3b8; font-size: 0.85rem; line-height: 1.5;
                        margin: 0.5rem 0 0 0; }

    .badge { display: inline-block; padding: 3px 12px; border-radius: 20px;
             font-size: 0.8rem; font-weight: 600; }
    .badge-green  { background: rgba(34,197,94,0.15); color: #22c55e; }
    .badge-yellow { background: rgba(234,179,8,0.15); color: #eab308; }
    .badge-red    { background: rgba(239,68,68,0.15); color: #ef4444; }

    .score-gauge {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-radius: 12px; padding: 1rem; text-align: center;
        border: 1px solid #334155;
    }
    .score-gauge .label { color: #94a3b8; font-size: 0.7rem;
                          text-transform: uppercase; letter-spacing: 1px; }
    .score-gauge .value { font-size: 2.2rem; font-weight: 700; margin: 0.2rem 0; }

    .playground-result {
        background: #0f172a; border-radius: 12px; padding: 1.5rem;
        border: 1px solid #334155; margin: 1rem 0;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600; font-size: 0.9rem; letter-spacing: 0.3px;
    }

    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Database helpers  (Statement Execution API — no extra package needed)
# ---------------------------------------------------------------------------

def _get_workspace_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def run_query(sql: str) -> pd.DataFrame:
    try:
        import requests as _req
        w = _get_workspace_client()
        auth = w.config.authenticate()
        host = w.config.host.rstrip("/")
        resp = _req.post(
            f"{host}/api/2.0/sql/statements",
            headers={**auth, "Content-Type": "application/json"},
            json={
                "warehouse_id": WAREHOUSE_ID,
                "statement": sql,
                "wait_timeout": "120s",
                "disposition": "INLINE",
                "format": "JSON_ARRAY",
            },
            timeout=130,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", {}).get("state", "")
        if status == "FAILED":
            msg = data.get("status", {}).get("error", {}).get("message", "Unknown SQL error")
            st.error(f"SQL error: {msg}")
            return pd.DataFrame()
        cols = [c["name"] for c in data.get("manifest", {}).get("schema", {}).get("columns", [])]
        rows = data.get("result", {}).get("data_array", [])
        return pd.DataFrame(rows, columns=cols)
    except Exception as exc:
        st.error(f"SQL error: {exc}")
        return pd.DataFrame()


def run_exec(sql: str):
    import requests as _req
    w = _get_workspace_client()
    auth = w.config.authenticate()
    host = w.config.host.rstrip("/")
    resp = _req.post(
        f"{host}/api/2.0/sql/statements",
        headers={**auth, "Content-Type": "application/json"},
        json={
            "warehouse_id": WAREHOUSE_ID,
            "statement": sql,
            "wait_timeout": "120s",
        },
        timeout=130,
    )
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(prompt: str, model: str = None) -> str:
    from databricks.sdk import WorkspaceClient
    import requests as _req

    if model is None:
        model = AVAILABLE_MODELS[0]
    w = WorkspaceClient()
    auth = w.config.authenticate()
    host = w.config.host.rstrip("/")
    resp = _req.post(
        f"{host}/serving-endpoints/{model}/invocations",
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


def evaluate_prompt(text: str, model: str = None) -> dict:
    instruction = (
        "You are a prompt engineering expert. Evaluate the following prompt on "
        "a scale of 0-100 for each dimension. Return ONLY a valid JSON object "
        "(no markdown, no code fences, no explanation) with these exact keys: "
        "overall_score (int), clarity_score (int), specificity_score (int), "
        "context_score (int), structure_score (int), improvement_suggestion "
        "(string, 1-2 sentences), improved_prompt (string, rewritten better "
        'version).\n\nPrompt to evaluate: "' + text + '"'
    )
    raw = call_llm(instruction, model)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Data loader
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
# Visual helpers
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


def _hex_to_rgba(hex_color, alpha=0.13):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_radar(scores, labels=None, color="#6c63ff", height=280):
    if labels is None:
        labels = ["Clarity", "Specificity", "Context", "Structure"]
    vals = list(scores) + [scores[0]]
    theta = labels + [labels[0]]
    fig = go.Figure(data=go.Scatterpolar(
        r=vals, theta=theta, fill="toself",
        fillcolor=_hex_to_rgba(color, 0.2),
        line=dict(color=color, width=2),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            showticklabels=False, gridcolor="#334155"),
            angularaxis=dict(gridcolor="#334155", linecolor="#334155"),
            bgcolor="#0e1117",
        ),
        showlegend=False, paper_bgcolor="#0e1117",
        font=dict(color="#e2e8f0"),
        height=height, margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


# ===========================  HEADER  =====================================
st.markdown("""
<div class="main-header">
    <h1>Jayd &mdash; Prompt Intelligence Platform</h1>
    <p>Score, improve, and execute prompts at enterprise scale &mdash; powered by Databricks Foundation Models</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Playground", "Analytics", "Template Library", "Live Feed", "Prompt Lab"
])

# ===========================  TAB 1 — PLAYGROUND  =========================
with tab1:
    st.markdown("### Prompt Playground")
    st.caption(
        "Enter any prompt, pick an LLM, and watch it get evaluated, improved, "
        "and executed in real time."
    )

    pg_col1, pg_col2 = st.columns([3, 1])
    with pg_col2:
        selected_model = st.selectbox(
            "LLM Model",
            AVAILABLE_MODELS,
            index=0,
            help="Choose which Foundation Model to use for evaluation and execution.",
        )

    with pg_col1:
        user_prompt = st.text_area(
            "Your Prompt",
            height=120,
            placeholder="Type or paste a prompt to evaluate... e.g. 'Write a marketing email for our new product launch'",
            key="pg_prompt",
        )

    run_btn = st.button("Evaluate & Run", type="primary", use_container_width=True)

    if run_btn and user_prompt.strip():
        # --- Step 1: Evaluate ---
        with st.status("Evaluating your prompt...", expanded=True) as status:
            st.write("Scoring on 4 dimensions with " + selected_model + "...")
            try:
                ev = evaluate_prompt(user_prompt.strip(), selected_model)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.stop()

            o = ev.get("overall_score", 0)
            cl = ev.get("clarity_score", 0)
            sp = ev.get("specificity_score", 0)
            cx = ev.get("context_score", 0)
            sr = ev.get("structure_score", 0)
            sug = ev.get("improvement_suggestion", "N/A")
            imp = ev.get("improved_prompt", user_prompt)

            status.update(label="Evaluation complete!", state="complete")

        # --- Scores display ---
        st.markdown("---")
        st.markdown("#### Scores")
        s1, s2, s3, s4, s5 = st.columns(5)
        for col, (lbl, val) in zip(
            [s1, s2, s3, s4, s5],
            [("Overall", o), ("Clarity", cl), ("Specificity", sp),
             ("Context", cx), ("Structure", sr)],
        ):
            col.markdown(
                f'<div class="kpi-card"><h3>{lbl}</h3>'
                f'<h1 style="color:{_color(val)}">{val}</h1></div>',
                unsafe_allow_html=True,
            )

        # --- Radar + suggestion side by side ---
        rc1, rc2 = st.columns([1, 1])
        with rc1:
            fig = make_radar([cl, sp, cx, sr])
            st.plotly_chart(fig, use_container_width=True)
        with rc2:
            st.markdown("##### Improvement Suggestion")
            st.info(sug)
            st.markdown("##### Improved Prompt")
            st.success(imp)

        # --- Step 2: Execute improved prompt ---
        st.markdown("---")
        st.markdown("#### LLM Response (Improved Prompt)")
        with st.spinner(f"Executing improved prompt on {selected_model}..."):
            try:
                llm_response = call_llm(imp, selected_model)
                st.markdown(
                    f'<div class="playground-result">{llm_response}</div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Execution failed: {exc}")

        # --- Persist to DB ---
        try:
            pid = f"APP-{uuid.uuid4().hex[:8].upper()}"
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            safe = user_prompt.replace("'", "''")
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

    elif run_btn:
        st.warning("Please enter a prompt first.")


# ===========================  TAB 2 — ANALYTICS  =========================
with tab2:
    st.markdown("### Prompt Quality Analytics")
    df = load_silver()

    if df.empty:
        st.warning("No evaluated prompts yet. Run the pipeline first.")
    else:
        total = len(df)
        scored = df["overall_score"].dropna()
        avg = scored.mean() if len(scored) > 0 else 0
        pct70 = (scored >= 70).sum() / max(len(scored), 1) * 100
        dept_avg = df.groupby("department")["overall_score"].mean()
        top_d = dept_avg.idxmax() if len(dept_avg) > 0 else "N/A"

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(
            f'<div class="kpi-card"><h3>Total Prompts</h3>'
            f'<h1 style="color:#a78bfa">{total}</h1></div>',
            unsafe_allow_html=True,
        )
        k2.markdown(
            f'<div class="kpi-card"><h3>Avg Score</h3>'
            f'<h1 class="{_cls(avg)}">{avg:.0f}</h1></div>',
            unsafe_allow_html=True,
        )
        k3.markdown(
            f'<div class="kpi-card"><h3>% Above 70</h3>'
            f'<h1 class="{_cls(pct70)}">{pct70:.0f}%</h1></div>',
            unsafe_allow_html=True,
        )
        k4.markdown(
            f'<div class="kpi-card"><h3>Top Department</h3>'
            f'<h1 style="color:#a78bfa;font-size:1.2rem">{top_d}</h1></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Score Distribution")
            fig = px.histogram(
                df, x="overall_score", nbins=20,
                color_discrete_sequence=["#6c63ff"],
            )
            fig.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="#334155", title="Overall Score"),
                yaxis=dict(gridcolor="#334155", title="Count"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("##### Avg Score by Category")
            ca = (
                df.groupby("category")["overall_score"]
                .mean()
                .reset_index()
                .sort_values("overall_score")
            )
            fig = px.bar(
                ca, x="overall_score", y="category", orientation="h",
                color="overall_score",
                color_continuous_scale=["#ef4444", "#eab308", "#22c55e"],
            )
            fig.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"),
                showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Score Dimensions by Department")
        dims = ["clarity_score", "specificity_score", "context_score", "structure_score"]
        ds = df.groupby("department")[dims].mean()
        colors = ["#6c63ff", "#22c55e", "#eab308", "#ef4444", "#3b82f6", "#ec4899"]
        fig = go.Figure()
        for i, (dept, row) in enumerate(ds.iterrows()):
            vals = [row[d] for d in dims] + [row[dims[0]]]
            c = colors[i % len(colors)]
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=["Clarity", "Specificity", "Context", "Structure", "Clarity"],
                fill="toself", name=dept,
                line=dict(color=c),
                fillcolor=_hex_to_rgba(c, 0.13),
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155"),
                angularaxis=dict(gridcolor="#334155"),
                bgcolor="#0e1117",
            ),
            paper_bgcolor="#0e1117", font=dict(color="#e2e8f0"), height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Bottom 10 — Improvement Opportunities")
        bot_cols = ["prompt_id", "prompt_text", "department", "category",
                    "overall_score", "improvement_suggestion"]
        bot_cols = [c for c in bot_cols if c in df.columns]
        bot = df.nsmallest(10, "overall_score")[bot_cols]
        st.dataframe(bot, use_container_width=True, hide_index=True)


# ===========================  TAB 3 — TEMPLATE LIBRARY  ===================
with tab3:
    st.markdown("### Template Library")
    st.caption(
        "15 curated, production-ready prompt templates across Business Units. "
        "Click 'Use in Playground' to evaluate and run any template."
    )

    # Group templates by BU
    bus = sorted(set(t["bu"] for t in PROMPT_TEMPLATES))
    filter_bu = st.selectbox("Filter by Business Unit", ["All"] + bus, key="tpl_filter")

    templates = PROMPT_TEMPLATES
    if filter_bu != "All":
        templates = [t for t in templates if t["bu"] == filter_bu]

    for idx, tpl in enumerate(templates):
        st.markdown(
            f'<div class="template-card">'
            f'<span class="bu-tag">{tpl["bu"]}</span>'
            f'<h4>{tpl["title"]}</h4>'
            f'<p>{tpl["prompt"][:200]}...</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        tc1, tc2 = st.columns([4, 1])
        with tc1:
            st.text_area(
                "Full Prompt",
                tpl["prompt"],
                height=100,
                key=f"tpl_{idx}",
                disabled=True,
            )
        with tc2:
            if st.button("Use in Playground", key=f"use_tpl_{idx}", type="secondary"):
                st.session_state["pg_prompt"] = tpl["prompt"]
                st.toast(f"Template loaded! Switch to the Playground tab.")
        st.markdown("")

    # Also show top DB prompts if available
    st.markdown("---")
    st.markdown("##### Top Evaluated Prompts from Database")
    df = load_silver()
    if not df.empty:
        top = df[df["overall_score"] >= 70].nlargest(10, "overall_score")
        if top.empty:
            st.info("No prompts scored 70+ yet in the database.")
        else:
            for _, r in top.iterrows():
                s = r["overall_score"]
                st.markdown(
                    f'<div class="template-card">'
                    f'<span class="bu-tag">{r.get("department", "N/A")}</span>'
                    f'&nbsp;<span class="badge {_badge(s)}">{int(s)}/100</span>'
                    f'<p style="color:#e2e8f0;margin-top:0.5rem">{str(r.get("prompt_text",""))[:200]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                a, b = st.columns(2)
                a.text_area(
                    "Original", str(r.get("prompt_text", "")), height=80,
                    key=f"dbo_{r['prompt_id']}", disabled=True,
                )
                b.text_area(
                    "Improved", str(r.get("improved_prompt", "N/A")), height=80,
                    key=f"dbi_{r['prompt_id']}", disabled=True,
                )
    else:
        st.info("No database prompts loaded yet.")


# ===========================  TAB 4 — LIVE FEED  =========================
with tab4:
    st.markdown("### Live Prompt Feed")
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
        if ss != "All":
            fd = fd[fd["source"] == ss]
        if sc != "All":
            fd = fd[fd["category"] == sc]
        if sd != "All":
            fd = fd[fd["department"] == sd]

        st.markdown(f"**Showing {len(fd)} prompts**")

        def _cs(v):
            try:
                v = float(v)
            except Exception:
                return ""
            if v >= 70:
                return "background-color:rgba(34,197,94,.15);color:#22c55e"
            if v >= 40:
                return "background-color:rgba(234,179,8,.15);color:#eab308"
            return "background-color:rgba(239,68,68,.15);color:#ef4444"

        show = ["prompt_id", "prompt_text", "department", "category", "source",
                "overall_score", "clarity_score", "specificity_score",
                "context_score", "structure_score"]
        show = [c for c in show if c in fd.columns]
        score_cols = [c for c in ["overall_score", "clarity_score",
                                   "specificity_score", "context_score",
                                   "structure_score"] if c in fd.columns]
        styled = fd[show].style.map(_cs, subset=score_cols)
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

        st.markdown("##### Prompt Details")
        if not fd.empty:
            sel_id = st.selectbox("Select prompt", fd["prompt_id"].tolist())
            if sel_id:
                r = fd[fd["prompt_id"] == sel_id].iloc[0]
                with st.expander(f"Details -- {sel_id}", expanded=True):
                    st.markdown(f"**Prompt:** {r['prompt_text']}")
                    st.markdown(
                        f"**Overall:** {r['overall_score']} | "
                        f"Clarity {r['clarity_score']} | "
                        f"Specificity {r['specificity_score']} | "
                        f"Context {r['context_score']} | "
                        f"Structure {r['structure_score']}"
                    )
                    st.markdown(f"**Suggestion:** {r.get('improvement_suggestion', '')}")
                    st.markdown(f"**Improved:** {r.get('improved_prompt', '')}")


# ===========================  TAB 5 — PROMPT LAB (Chat)  ==================
with tab5:
    st.markdown("### Prompt Lab (Chat)")
    st.caption("Chat-style interface for iterative prompt refinement.")

    lab_model = st.selectbox(
        "Model", AVAILABLE_MODELS, index=0, key="lab_model"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    prompt = st.chat_input("Enter your prompt to evaluate...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Evaluating..."):
                try:
                    ev = evaluate_prompt(prompt, lab_model)
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
                            f'<div class="kpi-card"><h3>{lbl}</h3>'
                            f'<h1 style="color:{_color(val)}">{val}</h1></div>',
                            unsafe_allow_html=True,
                        )

                    fig = make_radar([cl, sp, cx, sr])
                    st.plotly_chart(fig, use_container_width=True)

                    sug = ev.get("improvement_suggestion", "N/A")
                    imp = ev.get("improved_prompt", "N/A")
                    st.markdown("**Suggestion:**")
                    st.info(sug)
                    st.markdown("**Improved Prompt:**")
                    st.success(imp)
                    st.session_state["last_improved"] = imp
                    st.session_state["last_lab_model"] = lab_model

                    md = (f"**Score: {o}/100** | Clarity {cl} | Specificity {sp}"
                          f" | Context {cx} | Structure {sr}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": md})

                except Exception as exc:
                    st.error(f"Evaluation error: {exc}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {exc}"})

    if st.session_state.get("last_improved"):
        if st.button("Execute Improved Prompt", type="primary", key="exec_btn"):
            m = st.session_state.get("last_lab_model", AVAILABLE_MODELS[0])
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    try:
                        resp = call_llm(st.session_state["last_improved"], m)
                        st.markdown("**LLM Response:**")
                        st.markdown(resp)
                        st.session_state.messages.append(
                            {"role": "assistant",
                             "content": f"**Response:**\n\n{resp}"})
                    except Exception as exc:
                        st.error(f"Execution error: {exc}")
