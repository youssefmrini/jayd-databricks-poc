import os, json, uuid, re
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Config ───────────────────────────────────────────────────────────────────
CATALOG = "main"
SCHEMA = "jayd_poc"
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "d85fb7ed40320552")

AVAILABLE_MODELS = [
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-meta-llama-3-1-405b-instruct",
    "databricks-dbrx-instruct",
    "databricks-mixtral-8x7b-instruct",
]

# ─── Jayd.ai Brand Palette ───────────────────────────────────────────────────
PURPLE = "#7c3bed"
PURPLE_LIGHT = "#b375f0"
PINK = "#ff337e"
ORANGE = "#ff8c42"
BG = "#0a0819"
BG_CARD = "#13102a"
BORDER = "#2a2448"
TEXT = "#f0edf7"
TEXT_MUTED = "#8b83a8"
GREEN = "#22c55e"
AMBER = "#f59e0b"
RED = "#ef4444"
CHART_COLORS = [PURPLE, PINK, ORANGE, GREEN, "#3b82f6", AMBER, "#06b6d4", PURPLE_LIGHT, "#ec4899", "#10b981"]

# ─── 15 Prompt Templates ─────────────────────────────────────────────────────
PROMPT_TEMPLATES = [
    {"bu": "Marketing", "title": "Campaign Performance Deep-Dive", "prompt": "Analyze the Q1 2026 digital marketing campaign for our B2B SaaS product. The campaign ran across LinkedIn Ads, Google Search, and email nurture sequences targeting VP-level decision-makers in financial services. Total spend was $45,000. Results: 12,400 impressions, 890 clicks, 67 MQLs, 23 SQLs, and 4 closed-won deals worth $320K ARR. Calculate CAC, ROAS, and conversion rates at each funnel stage. Compare against industry benchmarks for B2B SaaS. Identify the highest-performing channel and recommend budget reallocation for Q2."},
    {"bu": "Sales", "title": "Competitive Deal Strategy", "prompt": "We are competing against Snowflake and Databricks in a $2M enterprise deal with a Fortune 500 retail company. The customer currently uses Hadoop on-prem with Hive and Spark. Their key requirements are: real-time inventory analytics, ML-based demand forecasting, and a unified governance layer across 3 cloud regions. Our platform strengths are lakehouse architecture and built-in ML. Draft a competitive positioning document with: (1) a SWOT analysis for each vendor, (2) three killer differentiators to highlight in our next executive briefing, and (3) potential objections the customer might raise with suggested rebuttals."},
    {"bu": "Engineering", "title": "Code Review & Refactoring Plan", "prompt": "Review the following Python data pipeline that processes 50M rows daily from Kafka into Delta Lake. The pipeline currently takes 4.2 hours to complete and occasionally fails with OOM errors on the executor nodes. Identify: (1) performance bottlenecks such as unnecessary shuffles, suboptimal partitioning, or missing Z-ordering, (2) reliability gaps including missing retry logic, dead-letter queues, or schema evolution handling, (3) code quality issues like hardcoded values, missing type hints, or absent error handling. Provide a prioritized refactoring plan with estimated effort for each improvement."},
    {"bu": "Human Resources", "title": "Job Description Generator", "prompt": "Create a compelling job description for a Senior Machine Learning Engineer role at a Series C fintech startup (450 employees, $120M funding). The role sits in the Risk & Fraud Detection team and requires 5+ years of experience with production ML systems. Tech stack: Python, PySpark, MLflow, Databricks, and AWS. Include: role summary, key responsibilities (6-8 bullets), required qualifications, preferred qualifications, compensation range ($180K-$240K + equity), and a section on company culture emphasizing remote-first, learning budget, and diversity commitment."},
    {"bu": "Finance", "title": "Financial Forecasting Analysis", "prompt": "Build a 12-month rolling revenue forecast for our SaaS business using the following inputs: current ARR of $18.5M, net revenue retention of 112%, gross churn of 8% annually, average new ACV of $45K, current pipeline of $6.2M (weighted), sales cycle of 90 days, and seasonal patterns showing Q4 accounting for 35% of new bookings. Model three scenarios: base case, optimistic with planned product launch boosting win rates by 15%, conservative with economic downturn extending sales cycles by 30%."},
    {"bu": "Legal", "title": "Contract Clause Review", "prompt": "Review the following enterprise SaaS Master Service Agreement for a healthcare customer subject to HIPAA regulations. Flag any clauses that: (1) create unlimited liability exposure, (2) contain data residency requirements conflicting with multi-region cloud, (3) impose SLA penalties exceeding 99.9% uptime standard, (4) include non-standard IP assignment provisions, (5) require BAA terms beyond standard HIPAA. For each flagged clause provide risk level, plain English explanation, and suggested counter-language."},
    {"bu": "Product", "title": "Feature Prioritization Framework", "prompt": "We have 12 feature requests in our product backlog for Q2 2026. Using the RICE scoring framework (Reach, Impact, Confidence, Effort), prioritize: (1) AI-powered auto-complete for SQL editor, (2) native Git integration for notebooks, (3) RBAC for dashboards, (4) real-time collaboration on queries, (5) automated data quality monitoring, (6) Slack integration for alerts, (7) custom visualization builder, (8) API rate limiting dashboard, (9) multi-language support, (10) SSO with Okta/Azure AD, (11) data catalog search improvements, (12) mobile-responsive dashboard viewer."},
    {"bu": "Data & Analytics", "title": "Data Quality Assessment", "prompt": "Perform a comprehensive data quality assessment on our customer_360 table (2.3M rows, 47 columns) sourced from Salesforce CRM, Stripe billing, Mixpanel analytics, and Zendesk support. Check: (1) completeness — NULLs per column, flag >5%, (2) consistency — conflicting values across sources, (3) timeliness — records stale >7 days, (4) uniqueness — duplicate customers by email/domain, (5) accuracy — revenue figures deviating >20% from Stripe source. Output a data quality scorecard with red/amber/green ratings and remediation steps."},
    {"bu": "Customer Support", "title": "Escalation Response Template", "prompt": "A P1 enterprise customer (ARR $450K, contract renewing in 60 days) escalated: production ETL pipeline failing intermittently for 72 hours, stale BI dashboards. VP of Data emailed our CEO. Root cause: memory leak in platform update v3.2.1 affecting clusters >256GB RAM. Draft: (1) immediate acknowledgment email, (2) technical remediation plan with ETA, (3) goodwill gesture proposal, and (4) internal post-mortem action items."},
    {"bu": "Operations", "title": "Process Optimization Audit", "prompt": "Audit our order-to-cash process for a manufacturing company ($200M revenue). Current: Sales quote (2 days) -> Legal review (5 days) -> Finance credit (3 days) -> Ops provisioning (4 days) -> CS onboarding (10 days). Total: 24 days vs 12-day benchmark. Identify bottlenecks, parallelization opportunities, automation with existing tools (Salesforce, DocuSign, NetSuite), quick wins (30 days) vs strategic improvements (90 days). Target: 14 days."},
    {"bu": "Design", "title": "UX Audit & Recommendations", "prompt": "Conduct a UX audit of our B2B analytics dashboard based on 20 user interviews: (1) 65% can't find export button in 30s, (2) filter panel needs 8 clicks for date+dept+metric, (3) tooltips truncate at 50 chars, (4) color palette fails WCAG AA contrast, (5) mobile 22% usage but layout breaks <768px, (6) 'compare two metrics' takes 4.2min vs 1min target. For each finding provide severity, affected personas, design recommendation with wireframe description, and expected impact."},
    {"bu": "Research & Development", "title": "Research Synthesis Report", "prompt": "Synthesize latest research (2024-2026) on RAG for enterprise knowledge management: (1) chunking strategies comparison and retrieval accuracy impact, (2) vector database benchmarks (Pinecone vs Weaviate vs ChromaDB vs Databricks Vector Search) for >10M documents, (3) hybrid search combining dense and sparse retrieval, (4) evaluation frameworks (RAGAS, DeepEval), (5) production challenges: hallucination rates, latency, cost optimization. Output structured literature review with comparison matrix and recommended architecture."},
    {"bu": "IT & Infrastructure", "title": "Incident Response Runbook", "prompt": "Create a Sev-1 database outage runbook for production Databricks workspace (200+ engineers, 45 customer dashboards). Include: (1) triage checklist (first 15 min): paging, checks, comms templates, (2) diagnostics: cluster health, driver logs, warehouse queue, Unity Catalog metastore, network, (3) common root causes and fixes: credential expiry, quota exceeded, storage mounts, IAM changes, (4) escalation matrix with response times, (5) post-incident review template. Format as step-by-step for L2 on-call at 3 AM."},
    {"bu": "Executive Leadership", "title": "Board Presentation Briefing", "prompt": "Prepare Q1 2026 board briefing for Series B SaaS ($32M ARR, 180 employees, 340 customers). Metrics: ARR growth 45% YoY (target 50%), NDR 118%, gross margin 72%, burn multiple 1.8x, CAC payback 14 months, logo churn 12% (up from 9%). New: 3 Fortune 500 logos, AI assistant (28% adoption in 6 weeks). Hired VP Eng + VP Marketing. Challenges: longer mid-market cycles, cloud costs +18% QoQ. Structure: TL;DR, financial vs plan, strategic wins, concerns with mitigations, Q2 priorities."},
    {"bu": "Compliance & Risk", "title": "Regulatory Impact Assessment", "prompt": "Assess EU AI Act (effective Aug 2025) impact on our ML credit scoring product used by 12 European banks. System uses: gradient boosting on 5yr transaction data, real-time features from 200+ variables including demographics, automated decisions for loans <EUR 50K. Evaluate: risk category (likely high-risk Annex III), compliance requirements (transparency, human oversight, data governance, bias testing), gap analysis, estimated cost and timeline, action plan with milestones. Reference specific EU AI Act articles."},
]

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Jayd — Prompt Intelligence", layout="wide", page_icon="✦")

# ─── CSS (Jayd.ai branded) ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
.stApp {
    background: linear-gradient(180deg, #0a0819 0%, #110e24 50%, #0d0a1e 100%);
}
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Poppins', sans-serif !important;
    color: #f0edf7 !important;
}
p, span, label, div {
    font-family: 'Poppins', sans-serif;
}

/* ── Header ── */
.jayd-header {
    background: linear-gradient(135deg, #7c3bed 0%, #a65eed 40%, #ff337e 100%);
    padding: 2.8rem 3rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(124, 59, 237, 0.3);
}
.jayd-header::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -15%;
    width: 50%;
    height: 180%;
    background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.jayd-header h1 {
    color: #ffffff !important;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}
.jayd-header p {
    color: rgba(255,255,255,0.88);
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0.6rem 0 0 0;
    position: relative;
    z-index: 1;
}

/* ── KPI Cards ── */
.kpi-card {
    background: rgba(19, 16, 42, 0.85);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 1.6rem 1.2rem;
    text-align: center;
    border: 1px solid rgba(42, 36, 72, 0.8);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(124, 59, 237, 0.5);
    box-shadow: 0 12px 40px rgba(124, 59, 237, 0.2);
}
.kpi-card .label {
    color: #8b83a8;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-weight: 600;
    margin: 0;
}
.kpi-card .value {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0.4rem 0 0 0;
    line-height: 1;
}

/* ── Section Titles ── */
.sec-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f0edf7;
    margin: 1.8rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.2px;
}
.sec-title .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7c3bed, #ff337e);
    display: inline-block;
    flex-shrink: 0;
}

/* ── Divider ── */
.jayd-div {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a2448, #7c3bed40, #2a2448, transparent);
    margin: 2rem 0;
    border: none;
}

/* ── Template Cards ── */
.tpl-card {
    background: rgba(19, 16, 42, 0.65);
    border: 1px solid rgba(42, 36, 72, 0.8);
    border-left: 4px solid #7c3bed;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
}
.tpl-card:hover {
    border-left-color: #ff337e;
    box-shadow: 0 6px 24px rgba(124, 59, 237, 0.15);
    background: rgba(19, 16, 42, 0.85);
}
.tpl-card .bu-tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    background: rgba(124, 59, 237, 0.15);
    color: #b375f0;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.tpl-card h4 {
    color: #f0edf7 !important;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0.5rem 0 0.3rem 0;
}
.tpl-card p {
    color: #8b83a8;
    font-size: 0.82rem;
    line-height: 1.65;
    margin: 0;
}

/* ── Score Badges ── */
.sbadge {
    display: inline-flex;
    align-items: center;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
}
.sbadge.g { background: rgba(34,197,94,0.12); color: #22c55e; }
.sbadge.a { background: rgba(245,158,11,0.12); color: #f59e0b; }
.sbadge.r { background: rgba(239,68,68,0.12); color: #ef4444; }

/* ── Tabs ── */
div[data-testid="stTabs"] button {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.5px;
    color: #8b83a8 !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.3s ease;
    padding: 0.8rem 1.2rem;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f0edf7 !important;
    border-bottom-color: #7c3bed !important;
}
div[data-testid="stTabs"] button:hover {
    color: #b375f0 !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #7c3bed 0%, #ff337e 100%) !important;
    border: none !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600;
    border-radius: 12px;
    padding: 0.65rem 2rem;
    transition: all 0.3s ease;
    color: white !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 8px 25px rgba(124, 59, 237, 0.4);
    transform: translateY(-1px);
}
.stButton > button[kind="secondary"],
button[data-testid="stBaseButton-secondary"] {
    border: 1px solid #7c3bed !important;
    color: #b375f0 !important;
    background: transparent !important;
    border-radius: 12px;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600;
}

/* ── LLM Response Box ── */
.llm-box {
    background: rgba(19, 16, 42, 0.85);
    border: 1px solid rgba(42, 36, 72, 0.8);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin: 1rem 0;
    color: #f0edf7;
    font-size: 0.92rem;
    line-height: 1.75;
}

/* ── Chat ── */
.stChatMessage { border-radius: 16px !important; }

/* ── Chart wrapper ── */
.chart-wrap {
    background: rgba(19, 16, 42, 0.45);
    border: 1px solid rgba(42, 36, 72, 0.6);
    border-radius: 18px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

/* ── Spacer ── */
.spacer { margin-top: 1.5rem; }
.spacer-sm { margin-top: 0.8rem; }

/* ── Analytics info banner ── */
.info-banner {
    background: rgba(124, 59, 237, 0.08);
    border: 1px solid rgba(124, 59, 237, 0.25);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    color: #b375f0;
    font-size: 0.9rem;
    text-align: center;
    margin: 2rem 0;
}

/* ── Selectbox & inputs ── */
div[data-baseweb="select"] > div {
    border-radius: 12px !important;
    border-color: #2a2448 !important;
    background: rgba(19, 16, 42, 0.8) !important;
}
.stTextArea textarea {
    border-radius: 12px !important;
    border-color: #2a2448 !important;
    background: rgba(19, 16, 42, 0.6) !important;
    color: #f0edf7 !important;
    font-family: 'Poppins', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_workspace_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def run_query(sql: str) -> pd.DataFrame:
    """Run SQL via SDK statement_execution API."""
    try:
        w = _get_workspace_client()
        resp = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=sql,
            wait_timeout="120s",
        )
        state_str = str(getattr(resp.status, "state", ""))
        if "FAILED" in state_str:
            err = getattr(resp.status, "error", None)
            msg = getattr(err, "message", "Unknown error") if err else "Unknown error"
            st.error(f"Query failed: {msg}")
            return pd.DataFrame()
        if not resp.manifest or not resp.manifest.schema:
            return pd.DataFrame()
        cols = [c.name for c in resp.manifest.schema.columns]
        rows = []
        if resp.result and resp.result.data_array:
            rows = resp.result.data_array
        return pd.DataFrame(rows, columns=cols)
    except Exception as exc:
        st.error(f"Query error: {exc}")
        return pd.DataFrame()


def run_exec(sql: str):
    """Execute a write SQL statement."""
    w = _get_workspace_client()
    w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        wait_timeout="120s",
    )


def call_llm(prompt: str, model: str = None) -> str:
    import requests as _req
    w = _get_workspace_client()
    if model is None:
        model = AVAILABLE_MODELS[0]
    host = w.config.host.rstrip("/")
    hf = w.config.authenticate()
    auth_h = hf() if callable(hf) else (dict(hf) if hf else {})
    resp = _req.post(
        f"{host}/serving-endpoints/{model}/invocations",
        headers={**auth_h, "Content-Type": "application/json"},
        json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": 0.3},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def evaluate_prompt(text: str, model: str = None) -> dict:
    instruction = (
        "You are a prompt engineering expert. Evaluate the following prompt on "
        "a scale of 0-100 for each dimension. Return ONLY a valid JSON object "
        "(no markdown, no code fences) with keys: overall_score (int), "
        "clarity_score (int), specificity_score (int), context_score (int), "
        "structure_score (int), improvement_suggestion (string 1-2 sentences), "
        'improved_prompt (string).\n\nPrompt: "' + text + '"'
    )
    raw = call_llm(instruction, model).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1:
        raw = raw[s:e + 1]
    return json.loads(raw)


# ─── Data Loader ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_silver():
    df = run_query(f"SELECT * FROM {CATALOG}.{SCHEMA}.silver_evaluated_prompts ORDER BY evaluated_at DESC")
    for c in ["overall_score", "clarity_score", "specificity_score", "context_score", "structure_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ─── Visual Helpers ───────────────────────────────────────────────────────────

def _hex_to_rgba(hx, a=0.13):
    h = hx.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def _color(v):
    try: v = float(v)
    except: return "#8b83a8"
    return GREEN if v >= 70 else AMBER if v >= 40 else RED

def _badge_cls(v):
    try: v = float(v)
    except: return "a"
    return "g" if v >= 70 else "a" if v >= 40 else "r"

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Poppins, sans-serif", color=TEXT, size=12),
    margin=dict(l=50, r=30, t=30, b=50),
)

def _axis_style():
    return dict(gridcolor="rgba(42,36,72,0.4)", zerolinecolor=BORDER, tickfont=dict(color=TEXT_MUTED))

def make_radar(scores, labels=None, color=PURPLE, height=300):
    if labels is None:
        labels = ["Clarity", "Specificity", "Context", "Structure"]
    vals = list(scores) + [scores[0]]
    theta = labels + [labels[0]]
    fig = go.Figure(data=go.Scatterpolar(
        r=vals, theta=theta, fill="toself",
        fillcolor=_hex_to_rgba(color, 0.18),
        line=dict(color=color, width=2.5),
        marker=dict(size=6, color=color),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                           gridcolor="rgba(42,36,72,0.35)", tickfont=dict(color=TEXT_MUTED, size=10)),
            angularaxis=dict(gridcolor="rgba(42,36,72,0.35)", linecolor=BORDER,
                            tickfont=dict(color=TEXT, size=12, family="Poppins")),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False, height=height,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, family="Poppins"),
        margin=dict(l=60, r=60, t=30, b=30),
    )
    return fig


# ─── PII Detection & Redaction ────────────────────────────────────────────────

_PII_PATTERNS = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]', 'Email'),
    (r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b', '[PHONE REDACTED]', 'Phone number'),
    (r'\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{0,4}\s?\d{0,2}\b', '[IBAN REDACTED]', 'IBAN'),
    (r'\b(?:FR|DE|GB|IT|ES|NL|BE|AT|CH|LU)\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{0,4}\b', '[IBAN REDACTED]', 'IBAN'),
    (r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b', '[SSN REDACTED]', 'SSN'),
    (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CARD REDACTED]', 'Credit card'),
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP REDACTED]', 'IP address'),
    (r'\b(?:password|passwd|pwd|secret|api[_-]?key|token|access[_-]?key)\s*[:=]\s*\S+', '[SECRET REDACTED]', 'Secret/Key'),
]

def detect_pii(text: str) -> list:
    """Return list of (match, pii_type) tuples found in text."""
    found = []
    for pattern, _, pii_type in _PII_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            found.append((m.group(), pii_type))
    return found

def redact_pii(text: str) -> tuple:
    """Return (redacted_text, list_of_redactions)."""
    redactions = []
    result = text
    for pattern, replacement, pii_type in _PII_PATTERNS:
        matches = list(re.finditer(pattern, result, re.IGNORECASE))
        for m in reversed(matches):
            redactions.append((m.group(), pii_type))
            result = result[:m.start()] + replacement + result[m.end():]
    return result, redactions


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="jayd-header">
    <h1>Jayd &mdash; Prompt Intelligence Platform</h1>
    <p>Score, improve, and execute prompts at enterprise scale &mdash; powered by Databricks Foundation Models</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Playground  ", "  Analytics  ", "  Template Library  ", "  Live Feed  ", "  Prompt Lab  "
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-title"><span class="dot"></span>Prompt Playground</div>', unsafe_allow_html=True)
    st.caption("Enter any prompt, pick an LLM, and watch it get evaluated, improved, and executed in real time.")
    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    pg1, pg2 = st.columns([3, 1])
    with pg2:
        selected_model = st.selectbox("LLM Model", AVAILABLE_MODELS, index=0,
                                      help="Choose which Foundation Model to use.")
    with pg1:
        user_prompt = st.text_area("Your Prompt", height=120,
                                   placeholder="Type or paste a prompt to evaluate...", key="pg_prompt")

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
    run_btn = st.button("Evaluate & Run", type="primary", use_container_width=True)

    if run_btn and user_prompt.strip():
        # ── PII Scan on original prompt ──
        pii_found = detect_pii(user_prompt.strip())
        if pii_found:
            st.markdown(
                '<div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);'
                'border-radius:14px;padding:1.2rem 1.6rem;margin:1rem 0">'
                '<span style="color:#ef4444;font-weight:700;font-size:0.95rem">'
                'PII Detected in Your Prompt</span></div>',
                unsafe_allow_html=True)
            for match, pii_type in pii_found:
                st.markdown(f'<span style="color:#ef4444;font-size:0.85rem">  {pii_type}: <code>{match}</code></span>', unsafe_allow_html=True)
            redacted_prompt, _ = redact_pii(user_prompt.strip())
            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color:{TEXT_MUTED};font-size:0.85rem"><b>Redacted version:</b></div>', unsafe_allow_html=True)
            st.code(redacted_prompt, language=None)
            eval_text = redacted_prompt
        else:
            eval_text = user_prompt.strip()

        with st.status("Evaluating your prompt...", expanded=True) as status:
            st.write(f"Scoring on 4 dimensions with {selected_model}...")
            try:
                ev = evaluate_prompt(eval_text, selected_model)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.stop()
            o  = ev.get("overall_score", 0)
            cl = ev.get("clarity_score", 0)
            sp = ev.get("specificity_score", 0)
            cx = ev.get("context_score", 0)
            sr = ev.get("structure_score", 0)
            sug = ev.get("improvement_suggestion", "N/A")
            imp = ev.get("improved_prompt", eval_text)
            status.update(label="Evaluation complete!", state="complete")

        st.markdown('<div class="jayd-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title"><span class="dot"></span>Scores</div>', unsafe_allow_html=True)

        cols = st.columns(5)
        for col, (lbl, val) in zip(cols,
            [("Overall", o), ("Clarity", cl), ("Specificity", sp), ("Context", cx), ("Structure", sr)]):
            col.markdown(
                f'<div class="kpi-card"><div class="label">{lbl}</div>'
                f'<div class="value" style="color:{_color(val)}">{val}</div></div>',
                unsafe_allow_html=True)

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns([1, 1])
        with rc1:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(make_radar([cl, sp, cx, sr]), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with rc2:
            st.markdown('<div class="sec-title"><span class="dot"></span>Improvement Suggestion</div>', unsafe_allow_html=True)
            st.info(sug)
            st.markdown('<div class="sec-title"><span class="dot"></span>Improved Prompt</div>', unsafe_allow_html=True)
            st.success(imp)

        # ── PII scan on improved prompt ──
        imp_pii = detect_pii(imp)
        if imp_pii:
            st.markdown(
                '<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);'
                'border-radius:14px;padding:1rem 1.4rem;margin:0.8rem 0">'
                '<span style="color:#f59e0b;font-weight:600;font-size:0.85rem">'
                'PII also found in improved prompt — it will be auto-redacted before execution.</span></div>',
                unsafe_allow_html=True)
            imp, _ = redact_pii(imp)

        # ── Store improved prompt and show Accept button ──
        st.session_state["pg_improved"] = imp
        st.session_state["pg_model"] = selected_model
        st.session_state["pg_scores"] = {"o": o, "cl": cl, "sp": sp, "cx": cx, "sr": sr, "sug": sug}
        st.session_state["pg_original"] = eval_text

        st.markdown('<div class="jayd-div"></div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            accept = st.button("Accept Improved Prompt & Execute", type="primary", use_container_width=True, key="accept_btn")
        with a2:
            reject = st.button("Reject — Keep Original", type="secondary", use_container_width=True, key="reject_btn")

        if accept:
            exec_prompt = st.session_state["pg_improved"]
            st.markdown('<div class="sec-title"><span class="dot"></span>LLM Response (Improved Prompt)</div>', unsafe_allow_html=True)
            with st.spinner(f"Executing on {selected_model}..."):
                try:
                    llm_resp = call_llm(exec_prompt, selected_model)
                    st.markdown(f'<div class="llm-box">{llm_resp}</div>', unsafe_allow_html=True)
                except Exception as exc:
                    st.error(f"Execution failed: {exc}")
            # Persist
            try:
                pid = f"APP-{uuid.uuid4().hex[:8].upper()}"
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                safe = eval_text.replace("'", "''")
                safe_s = str(sug).replace("'", "''")
                safe_i = str(imp).replace("'", "''")
                run_exec(f"INSERT INTO {CATALOG}.{SCHEMA}.bronze_prompts VALUES('{pid}','{safe}','app_user','App','user_input','{now}','app','{now}')")
                run_exec(f"INSERT INTO {CATALOG}.{SCHEMA}.silver_evaluated_prompts VALUES('{pid}','{safe}','app_user','App','user_input','{now}','app',{o},{cl},{sp},{cx},{sr},'{safe_s}','{safe_i}','{now}')")
            except Exception:
                pass

        if reject:
            st.markdown('<div class="sec-title"><span class="dot"></span>LLM Response (Original Prompt)</div>', unsafe_allow_html=True)
            with st.spinner(f"Executing original on {selected_model}..."):
                try:
                    llm_resp = call_llm(eval_text, selected_model)
                    st.markdown(f'<div class="llm-box">{llm_resp}</div>', unsafe_allow_html=True)
                except Exception as exc:
                    st.error(f"Execution failed: {exc}")

    elif run_btn:
        st.warning("Please enter a prompt first.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title"><span class="dot"></span>Prompt Quality Analytics</div>', unsafe_allow_html=True)
    df = load_silver()

    if df.empty:
        st.markdown(
            '<div class="info-banner">No evaluated prompts yet. Run the data pipeline first to populate analytics.</div>',
            unsafe_allow_html=True)
    else:
        total = len(df)
        scored = df["overall_score"].dropna()
        avg = scored.mean() if len(scored) > 0 else 0
        pct70 = (scored >= 70).sum() / max(len(scored), 1) * 100
        dept_avg = df.groupby("department")["overall_score"].mean()
        top_d = dept_avg.idxmax() if len(dept_avg) > 0 else "N/A"
        low_dim = None
        for d in ["clarity_score", "specificity_score", "context_score", "structure_score"]:
            if d in df.columns:
                m = df[d].dropna().mean()
                if low_dim is None or m < low_dim[1]:
                    low_dim = (d.replace("_score", "").title(), m)

        # ── KPI Row ──
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="kpi-card"><div class="label">Total Prompts</div><div class="value" style="color:{PURPLE_LIGHT}">{total}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-card"><div class="label">Avg Score</div><div class="value" style="color:{_color(avg)}">{avg:.0f}</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-card"><div class="label">Above 70</div><div class="value" style="color:{_color(pct70)}">{pct70:.0f}%</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi-card"><div class="label">Top Department</div><div class="value" style="color:{PURPLE_LIGHT};font-size:1.1rem">{top_d}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # ── Row 2: Histogram + Category Bar ──
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title"><span class="dot"></span>Score Distribution</div>', unsafe_allow_html=True)
            fig = go.Figure(data=go.Histogram(
                x=scored, nbinsx=20,
                marker=dict(color=PURPLE, line=dict(color=PURPLE_LIGHT, width=1)),
                opacity=0.85,
            ))
            fig.update_layout(**PLOTLY_BASE, height=320,
                             xaxis=dict(title="Overall Score", **_axis_style()),
                             yaxis=dict(title="Count", **_axis_style()),
                             showlegend=False, bargap=0.08)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title"><span class="dot"></span>Avg Score by Category</div>', unsafe_allow_html=True)
            ca = df.groupby("category")["overall_score"].mean().reset_index().sort_values("overall_score")
            bar_colors = [_color(v) for v in ca["overall_score"]]
            fig = go.Figure(data=go.Bar(
                x=ca["overall_score"], y=ca["category"], orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                opacity=0.9,
            ))
            fig.update_layout(**PLOTLY_BASE, height=320, showlegend=False,
                             xaxis=dict(title="Avg Score", **_axis_style()),
                             yaxis=dict(**_axis_style()))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # ── Row 3: Radar + Dimension Comparison ──
        c3, c4 = st.columns(2)
        dims = ["clarity_score", "specificity_score", "context_score", "structure_score"]
        dim_labels = ["Clarity", "Specificity", "Context", "Structure"]

        with c3:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title"><span class="dot"></span>Department Radar</div>', unsafe_allow_html=True)
            ds = df.groupby("department")[dims].mean()
            fig = go.Figure()
            for i, (dept, row) in enumerate(ds.iterrows()):
                c = CHART_COLORS[i % len(CHART_COLORS)]
                vals = [row[d] for d in dims] + [row[dims[0]]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=dim_labels + [dim_labels[0]],
                    fill="toself", name=dept,
                    line=dict(color=c, width=2),
                    fillcolor=_hex_to_rgba(c, 0.1),
                ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                                   gridcolor="rgba(42,36,72,0.35)", tickfont=dict(color=TEXT_MUTED, size=10)),
                    angularaxis=dict(gridcolor="rgba(42,36,72,0.35)", tickfont=dict(color=TEXT, size=12)),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, family="Poppins"),
                height=380, legend=dict(font=dict(color=TEXT_MUTED, size=10)),
                margin=dict(l=60, r=60, t=30, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title"><span class="dot"></span>Score Dimensions Avg</div>', unsafe_allow_html=True)
            dim_means = [df[d].dropna().mean() for d in dims]
            dim_colors = [PURPLE, PINK, ORANGE, GREEN]
            fig = go.Figure(data=go.Bar(
                x=dim_labels, y=dim_means,
                marker=dict(color=dim_colors, line=dict(width=0)),
                text=[f"{v:.0f}" for v in dim_means],
                textposition="outside",
                textfont=dict(color=TEXT, family="Poppins", size=13),
            ))
            fig.update_layout(**PLOTLY_BASE, height=380, showlegend=False,
                             xaxis=dict(**_axis_style()), yaxis=dict(title="Avg Score", range=[0, 105], **_axis_style()))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # ── Row 4: Source Donut + Score Heatmap ──
        c5, c6 = st.columns(2)
        with c5:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title"><span class="dot"></span>Source Distribution</div>', unsafe_allow_html=True)
            if "source" in df.columns:
                src_counts = df["source"].value_counts().reset_index()
                src_counts.columns = ["source", "count"]
                fig = go.Figure(data=go.Pie(
                    labels=src_counts["source"], values=src_counts["count"],
                    hole=0.55, marker=dict(colors=[PURPLE, PINK, ORANGE, GREEN][:len(src_counts)],
                                           line=dict(color=BG, width=3)),
                    textfont=dict(color=TEXT, family="Poppins"),
                    textinfo="label+percent",
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, family="Poppins"),
                                 height=350, showlegend=True, legend=dict(font=dict(color=TEXT_MUTED)),
                                 margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No source column available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c6:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title"><span class="dot"></span>Dept × Dimension Heatmap</div>', unsafe_allow_html=True)
            hm = df.groupby("department")[dims].mean()
            hm.columns = dim_labels
            fig = go.Figure(data=go.Heatmap(
                z=hm.values, x=dim_labels, y=hm.index.tolist(),
                colorscale=[[0, "#1a1626"], [0.4, PURPLE], [0.7, PINK], [1, ORANGE]],
                showscale=True, colorbar=dict(tickfont=dict(color=TEXT_MUTED)),
                texttemplate="%{z:.0f}", textfont=dict(color=TEXT, size=11),
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, family="Poppins"),
                             height=350, margin=dict(l=120, r=20, t=20, b=50),
                             xaxis=dict(tickfont=dict(color=TEXT)), yaxis=dict(tickfont=dict(color=TEXT_MUTED, size=10)))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # ── Bottom 10 ──
        st.markdown('<div class="sec-title"><span class="dot"></span>Bottom 10 — Improvement Opportunities</div>', unsafe_allow_html=True)
        bot_cols = [c for c in ["prompt_id", "department", "category", "overall_score", "improvement_suggestion"] if c in df.columns]
        bot = df.nsmallest(10, "overall_score")[bot_cols]
        st.dataframe(bot, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — TEMPLATE LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title"><span class="dot"></span>Template Library</div>', unsafe_allow_html=True)
    st.caption("15 curated, production-ready prompt templates across Business Units.")
    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    bus = sorted(set(t["bu"] for t in PROMPT_TEMPLATES))
    filter_bu = st.selectbox("Filter by Business Unit", ["All"] + bus, key="tpl_filter")

    templates = PROMPT_TEMPLATES if filter_bu == "All" else [t for t in PROMPT_TEMPLATES if t["bu"] == filter_bu]

    # Grid layout — 2 columns
    for row_start in range(0, len(templates), 2):
        row_tpls = templates[row_start:row_start + 2]
        cols = st.columns(len(row_tpls))
        for col, tpl in zip(cols, row_tpls):
            idx = PROMPT_TEMPLATES.index(tpl)
            with col:
                st.markdown(
                    f'<div class="tpl-card">'
                    f'<span class="bu-tag">{tpl["bu"]}</span>'
                    f'<h4>{tpl["title"]}</h4>'
                    f'<p>{tpl["prompt"][:180]}...</p>'
                    f'</div>', unsafe_allow_html=True)
                with st.expander("View full prompt"):
                    st.markdown(f'<span style="color:#8b83a8;font-size:0.85rem;line-height:1.7">{tpl["prompt"]}</span>', unsafe_allow_html=True)
                if st.button("Use in Playground", key=f"use_{idx}", type="secondary"):
                    st.session_state["pg_prompt"] = tpl["prompt"]
                    st.toast("Template loaded! Switch to the Playground tab.")
                st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # DB top prompts
    st.markdown('<div class="jayd-div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title"><span class="dot"></span>Top Evaluated Prompts from Database</div>', unsafe_allow_html=True)
    df_tpl = load_silver()
    if not df_tpl.empty:
        top = df_tpl[df_tpl["overall_score"] >= 70].nlargest(10, "overall_score")
        if top.empty:
            st.markdown('<div class="info-banner">No prompts scored 70+ yet in the database.</div>', unsafe_allow_html=True)
        else:
            for _, r in top.iterrows():
                s = r["overall_score"]
                st.markdown(
                    f'<div class="tpl-card">'
                    f'<span class="bu-tag">{r.get("department", "N/A")}</span> '
                    f'<span class="sbadge {_badge_cls(s)}">{int(s)}/100</span>'
                    f'<p style="color:#d0ccdf;margin-top:0.5rem;font-size:0.85rem">{str(r.get("prompt_text",""))[:250]}</p>'
                    f'</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-banner">No database prompts loaded yet.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — LIVE FEED
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title"><span class="dot"></span>Live Prompt Feed</div>', unsafe_allow_html=True)
    df_feed = load_silver()

    if df_feed.empty:
        st.markdown('<div class="info-banner">No data yet. Run the pipeline to populate the feed.</div>', unsafe_allow_html=True)
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            srcs = ["All"] + sorted(df_feed["source"].dropna().unique().tolist())
            ss = st.selectbox("Source", srcs, key="feed_src")
        with f2:
            cats = ["All"] + sorted(df_feed["category"].dropna().unique().tolist())
            sc = st.selectbox("Category", cats, key="feed_cat")
        with f3:
            deps = ["All"] + sorted(df_feed["department"].dropna().unique().tolist())
            sd = st.selectbox("Department", deps, key="feed_dept")

        fd = df_feed.copy()
        if ss != "All": fd = fd[fd["source"] == ss]
        if sc != "All": fd = fd[fd["category"] == sc]
        if sd != "All": fd = fd[fd["department"] == sd]

        st.markdown(f'<div style="color:{TEXT_MUTED};font-size:0.85rem;margin:0.8rem 0">Showing <b style="color:{PURPLE_LIGHT}">{len(fd)}</b> prompts</div>', unsafe_allow_html=True)

        def _cell_color(v):
            try: v = float(v)
            except: return ""
            if v >= 70: return f"background-color:rgba(34,197,94,.1);color:{GREEN}"
            if v >= 40: return f"background-color:rgba(245,158,11,.1);color:{AMBER}"
            return f"background-color:rgba(239,68,68,.1);color:{RED}"

        show = [c for c in ["prompt_id", "department", "category", "source", "overall_score",
                            "clarity_score", "specificity_score", "context_score", "structure_score"] if c in fd.columns]
        score_cols = [c for c in ["overall_score", "clarity_score", "specificity_score", "context_score", "structure_score"] if c in fd.columns]
        styled = fd[show].style.map(_cell_color, subset=score_cols)
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title"><span class="dot"></span>Prompt Details</div>', unsafe_allow_html=True)
        if not fd.empty:
            sel_id = st.selectbox("Select prompt", fd["prompt_id"].tolist(), key="feed_sel")
            if sel_id:
                r = fd[fd["prompt_id"] == sel_id].iloc[0]
                with st.expander(f"Details — {sel_id}", expanded=True):
                    st.markdown(f"**Prompt:** {r['prompt_text']}")
                    st.markdown(
                        f"**Overall:** {r['overall_score']} | "
                        f"Clarity {r['clarity_score']} | Specificity {r['specificity_score']} | "
                        f"Context {r['context_score']} | Structure {r['structure_score']}")
                    st.markdown(f"**Suggestion:** {r.get('improvement_suggestion', '')}")
                    st.markdown(f"**Improved:** {r.get('improved_prompt', '')}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — PROMPT LAB (Chat)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-title"><span class="dot"></span>Prompt Lab</div>', unsafe_allow_html=True)
    st.caption("Chat-style interface for iterative prompt refinement.")
    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    lab_model = st.selectbox("Model", AVAILABLE_MODELS, index=0, key="lab_model")

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
                    o  = ev.get("overall_score", 0)
                    cl = ev.get("clarity_score", 0)
                    sp = ev.get("specificity_score", 0)
                    cx = ev.get("context_score", 0)
                    sr = ev.get("structure_score", 0)

                    cols = st.columns(5)
                    for c, (lbl, val) in zip(cols, [("Overall", o), ("Clarity", cl), ("Specificity", sp), ("Context", cx), ("Structure", sr)]):
                        c.markdown(f'<div class="kpi-card"><div class="label">{lbl}</div><div class="value" style="color:{_color(val)}">{val}</div></div>', unsafe_allow_html=True)

                    st.plotly_chart(make_radar([cl, sp, cx, sr]), use_container_width=True)

                    sug = ev.get("improvement_suggestion", "N/A")
                    imp = ev.get("improved_prompt", "N/A")
                    st.markdown(f'<div class="sec-title"><span class="dot"></span>Suggestion</div>', unsafe_allow_html=True)
                    st.info(sug)
                    st.markdown(f'<div class="sec-title"><span class="dot"></span>Improved Prompt</div>', unsafe_allow_html=True)
                    st.success(imp)
                    st.session_state["last_improved"] = imp
                    st.session_state["last_lab_model"] = lab_model

                    md = f"**Score: {o}/100** | Clarity {cl} | Specificity {sp} | Context {cx} | Structure {sr}"
                    st.session_state.messages.append({"role": "assistant", "content": md})
                except Exception as exc:
                    st.error(f"Evaluation error: {exc}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {exc}"})

    if st.session_state.get("last_improved"):
        if st.button("Execute Improved Prompt", type="primary", key="exec_btn"):
            m = st.session_state.get("last_lab_model", AVAILABLE_MODELS[0])
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    try:
                        resp = call_llm(st.session_state["last_improved"], m)
                        st.markdown(f'<div class="llm-box">{resp}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": f"**Response:**\n\n{resp}"})
                    except Exception as exc:
                        st.error(f"Execution error: {exc}")
