"""
Bharat Horizon Bank - Home Loan Assessment System
===================================================
A 4-screen Streamlit MVP:
  1. Loan Application Form
  2. AI Assessment (Logistic Regression + explainability)
  3. RAG Explainability Chatbot (OpenRouter LLM)
  4. Human-in-the-Loop Final Review

Author : College Project (Educational Purposes Only)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

# ============================================================
# 0. PAGE CONFIG & GLOBAL STYLING
# ============================================================

st.set_page_config(
    page_title="Bharat Horizon Bank - Home Loan Assessment",
    page_icon="\U0001F3E0",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.85);
    --bg-glass: rgba(255, 255, 255, 0.04);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(99, 102, 241, 0.3);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --accent-amber: #f59e0b;
    --gradient-hero: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    --gradient-success: linear-gradient(135deg, #059669 0%, #10b981 100%);
    --gradient-danger: linear-gradient(135deg, #dc2626 0%, #f43f5e 100%);
    --shadow-glow: 0 0 30px rgba(99, 102, 241, 0.15);
}
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] .stRadio > label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.75rem;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    padding: 0.65rem 1rem !important;
    border-radius: 10px !important;
    margin-bottom: 4px !important;
    transition: all 0.25s ease !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
    background: var(--bg-glass) !important;
    border-color: var(--border-accent) !important;
}
[data-testid="stExpander"], .stAlert {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stMetric"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { font-weight: 700 !important; }
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.25s ease !important;
    border: 1px solid var(--border-subtle) !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-glow);
}
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatMessage"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    margin-bottom: 0.5rem !important;
}
.hero-title {
    background: var(--gradient-hero);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    font-size: 1.8rem;
    margin-bottom: 0.25rem;
}
.hero-subtitle { color: var(--text-secondary); font-size: 0.95rem; font-weight: 400; }
.decision-approved {
    background: var(--gradient-success); color: white; padding: 1.5rem 2rem;
    border-radius: 16px; text-align: center; font-size: 1.6rem; font-weight: 700;
    letter-spacing: 0.02em; box-shadow: 0 8px 30px rgba(16, 185, 129, 0.25);
}
.decision-rejected {
    background: var(--gradient-danger); color: white; padding: 1.5rem 2rem;
    border-radius: 16px; text-align: center; font-size: 1.6rem; font-weight: 700;
    letter-spacing: 0.02em; box-shadow: 0 8px 30px rgba(244, 63, 94, 0.25);
}
.glass-card {
    background: var(--bg-glass); border: 1px solid var(--border-subtle);
    border-radius: 16px; padding: 1.5rem; backdrop-filter: blur(12px); margin-bottom: 1rem;
}
.feature-bar-positive { height: 8px; border-radius: 4px; background: var(--gradient-success); }
.feature-bar-negative { height: 8px; border-radius: 4px; background: var(--gradient-danger); }
.styled-divider { height: 1px; background: var(--border-subtle); margin: 1.5rem 0; }
.summary-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.06));
    border: 1px solid var(--border-accent); border-radius: 16px; padding: 1.75rem;
}
.final-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.08));
    border: 2px solid var(--accent-emerald); border-radius: 16px; padding: 2rem; text-align: center;
}
.final-rejected {
    background: linear-gradient(135deg, rgba(244,63,94,0.12), rgba(220,38,38,0.08));
    border: 2px solid var(--accent-rose); border-radius: 16px; padding: 2rem; text-align: center;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 3px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# 1. MODEL TRAINING (cached, runs once on boot)
# ============================================================

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bharat_horizon_home_loan_dataset_v2.xlsx")
POLICY_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bharat_horizon_bank_home_loan_policy.md")

@st.cache_resource(show_spinner="Training Logistic Regression model on first boot...")
def load_model_and_scaler():
    df = pd.read_excel(DATASET_PATH, sheet_name="Loan_Dataset")
    if "default_timing_demo_only" in df.columns:
        df = df.drop(columns=["default_timing_demo_only"])
    df["loan_outcome"] = df["loan_outcome"].map({"Approved": 1, "Rejected": 0})
    df = pd.get_dummies(df, columns=["employment_type"], drop_first=True)
    X = df.drop(columns=["loan_outcome"])
    y = df["loan_outcome"]
    feature_names = list(X.columns)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42)
    scaler_obj = StandardScaler()
    X_train_scaled = scaler_obj.fit_transform(X_train)
    model_obj = LogisticRegression(random_state=42)
    model_obj.fit(X_train_scaled, y_train)
    return model_obj, scaler_obj, feature_names

@st.cache_data(show_spinner=False)
def load_policy_text():
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        return f.read()

model, scaler, FEATURE_NAMES = load_model_and_scaler()
policy_text = load_policy_text()

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def build_feature_vector(data):
    monthly_income = data["monthly_income_inr"]
    existing_emi   = data["existing_emi_inr"]
    loan_amount    = data["loan_amount_inr"]
    tenure_years   = data["loan_tenure_years"]
    annual_rate  = 0.085
    monthly_rate = annual_rate / 12
    n_months     = tenure_years * 12
    if monthly_rate > 0 and n_months > 0:
        proposed_emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** n_months) / (
            ((1 + monthly_rate) ** n_months) - 1)
    else:
        proposed_emi = loan_amount / max(n_months, 1)
    foir = (existing_emi + proposed_emi) / monthly_income if monthly_income > 0 else 999
    loan_to_annual_income = loan_amount / (monthly_income * 12) if monthly_income > 0 else 999
    data["proposed_emi"] = round(proposed_emi, 2)
    data["foir"] = round(foir, 4)
    data["loan_to_annual_income_ratio"] = round(loan_to_annual_income, 4)
    is_self_employed = 1 if data["employment_type"] == "Self-employed" else 0
    row = {
        "monthly_income_inr": data["monthly_income_inr"],
        "credit_score": data["credit_score"],
        "loan_amount_inr": data["loan_amount_inr"],
        "loan_tenure_years": data["loan_tenure_years"],
        "existing_emi_inr": data["existing_emi_inr"],
        "employment_years": data["employment_years"],
        "past_default": data["past_default"],
        "foir": data["foir"],
        "loan_to_annual_income_ratio": data["loan_to_annual_income_ratio"],
        "employment_type_Self-employed": is_self_employed,
    }
    return pd.DataFrame([row])[FEATURE_NAMES]

def get_openrouter_client():
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def chat_with_llm(messages):
    client = get_openrouter_client()
    try:
        resp = client.chat.completions.create(
            model="google/gemini-2.5-flash", messages=messages,
            temperature=0.4, max_tokens=1024)
        return resp.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {e}"

# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown('<p class="hero-title">\U0001F3E0 Bharat Horizon</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Home Loan Assessment System</p>', unsafe_allow_html=True)
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    screen = st.radio(
        "NAVIGATION",
        ["Loan Application", "AI Assessment", "Explainability Chat", "Human Review"],
        index=0,
        format_func=lambda x: {
            "Loan Application": "\U0001F4DD  Loan Application",
            "AI Assessment": "\U0001F916  AI Assessment",
            "Explainability Chat": "\U0001F4AC  Explainability Chat",
            "Human Review": "\U0001F464  Human Review",
        }[x],
    )
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    app_sidebar = st.session_state.get("application", None)
    if app_sidebar:
        status_val = app_sidebar.get("final_status", app_sidebar.get("ai_decision", "Pending"))
        if "Approv" in str(status_val):
            scolor = "var(--accent-emerald)"
        elif "Reject" in str(status_val):
            scolor = "var(--accent-rose)"
        else:
            scolor = "var(--accent-amber)"
        st.markdown(
            '<div class="glass-card" style="text-align:center;">'
            '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
            'color:var(--text-muted);margin-bottom:4px;">Application Status</div>'
            '<div style="font-size:1.1rem;font-weight:700;color:' + scolor + ';">'
            '\u25CF ' + str(status_val) + '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="glass-card" style="text-align:center;">'
            '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
            'color:var(--text-muted);margin-bottom:4px;">Application Status</div>'
            '<div style="font-size:1.1rem;font-weight:700;color:var(--accent-amber);">'
            '\u25CF No Application</div></div>',
            unsafe_allow_html=True,
        )

# ============================================================
# 4. SCREEN 1 - LOAN APPLICATION FORM
# ============================================================

if screen == "Loan Application":
    st.markdown(
        '<div style="margin-bottom:1.5rem;">'
        '<p class="hero-title" style="font-size:1.5rem;">\U0001F4DD Loan Application Form</p>'
        '<p class="hero-subtitle">Fill in applicant details below to begin assessment.</p>'
        '</div>', unsafe_allow_html=True)
    st.warning(
        "**Disclaimer:** This application is for **educational purposes only**. "
        "It does not constitute an actual loan offer or financial advice.", icon="\U0001F4E2")
    with st.form("loan_form", clear_on_submit=False):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("##### \U0001F4B0 Financial Details")
            monthly_income = st.number_input("Monthly Income (INR)", min_value=10000, max_value=10000000, value=80000, step=5000)
            loan_amount = st.number_input("Requested Loan Amount (INR)", min_value=100000, max_value=500000000, value=3000000, step=100000)
            loan_tenure = st.number_input("Loan Tenure (Years)", min_value=1, max_value=30, value=20)
            existing_emi = st.number_input("Existing EMI / Monthly Debts (INR)", min_value=0, max_value=5000000, value=5000, step=1000)
        with col2:
            st.markdown("##### \U0001F9FE Credit & Employment")
            credit_score = st.number_input("Credit Score (300-900)", min_value=300, max_value=900, value=720, step=5)
            employment_years = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            employment_type = st.selectbox("Employment Type", ["Salaried", "Self-employed"])
            past_default = st.selectbox("Any Past Loan Default?", ["No", "Yes"], help="Defaults in the past 5 years")
        st.markdown("---")
        submitted = st.form_submit_button("\U0001F680  Submit Application", use_container_width=True)
    if submitted:
        app_data = {
            "monthly_income_inr": monthly_income, "credit_score": credit_score,
            "loan_amount_inr": loan_amount, "loan_tenure_years": loan_tenure,
            "existing_emi_inr": existing_emi, "employment_years": employment_years,
            "employment_type": employment_type, "past_default": 1 if past_default == "Yes" else 0,
        }
        feature_df = build_feature_vector(app_data)
        st.session_state["application"] = app_data
        st.session_state["feature_df"] = feature_df
        for key in ["ai_decision", "math_breakdown", "chat_history", "llm_summary", "ai_confidence", "contributions"]:
            st.session_state.pop(key, None)
        st.session_state["application"].pop("ai_decision", None)
        st.session_state["application"].pop("final_status", None)
        st.session_state["application"].pop("final_decision", None)
        st.success("\u2705 Application submitted successfully!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Proposed EMI", "INR {:,.0f}".format(app_data["proposed_emi"]))
        c2.metric("FOIR", "{:.1f}%".format(app_data["foir"] * 100))
        c3.metric("Loan-to-Income Ratio", "{:.2f}x".format(app_data["loan_to_annual_income_ratio"]))
        st.info("Navigate to **AI Assessment** in the sidebar to see the model's decision.")

# ============================================================
# 5. SCREEN 2 - AI ASSESSMENT
# ============================================================

elif screen == "AI Assessment":
    st.markdown(
        '<div style="margin-bottom:1.5rem;">'
        '<p class="hero-title" style="font-size:1.5rem;">\U0001F916 AI Assessment</p>'
        '<p class="hero-subtitle">Model prediction with full mathematical explainability.</p>'
        '</div>', unsafe_allow_html=True)
    if "application" not in st.session_state or "feature_df" not in st.session_state:
        st.info("Please submit a loan application on the **Loan Application** screen first.")
        st.stop()
    app_data = st.session_state["application"]
    feature_df = st.session_state["feature_df"]
    X_scaled = scaler.transform(feature_df)
    prob = model.predict_proba(X_scaled)[0]
    prediction = model.predict(X_scaled)[0]
    decision = "Approved" if prediction == 1 else "Rejected"
    confidence = prob[1] if prediction == 1 else prob[0]
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    scaled_vals = X_scaled[0]
    clist = []
    for i, fn in enumerate(FEATURE_NAMES):
        contrib = coefficients[i] * scaled_vals[i]
        clist.append({"Feature": fn, "Coefficient": round(coefficients[i], 4),
                      "Scaled Value": round(scaled_vals[i], 4), "Contribution": round(contrib, 4)})
    cdf = pd.DataFrame(clist).sort_values("Contribution", key=abs, ascending=False)
    math_lines = ["Intercept (bias): {:.4f}".format(intercept)]
    total = intercept
    for _, r in cdf.iterrows():
        s = "+" if r["Contribution"] >= 0 else ""
        math_lines.append("  {}: coeff={:.4f} x scaled={:.4f} -> {}{:.4f}".format(
            r["Feature"], r["Coefficient"], r["Scaled Value"], s, r["Contribution"]))
        total += r["Contribution"]
    math_lines.append("Total log-odds score: {:.4f}".format(total))
    math_lines.append("Probability of Approval: {:.1f}%".format(prob[1] * 100))
    math_breakdown = "\n".join(math_lines)
    app_data["ai_decision"] = decision
    st.session_state["ai_decision"] = decision
    st.session_state["ai_confidence"] = confidence
    st.session_state["math_breakdown"] = math_breakdown
    st.session_state["contributions"] = cdf.to_dict("records")
    if decision == "Approved":
        st.markdown(
            '<div class="decision-approved">\u2705 AI DECISION: APPROVED<br>'
            '<span style="font-size:0.85rem;font-weight:400;">Confidence: {:.1f}%</span></div>'.format(confidence*100),
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="decision-rejected">\u274C AI DECISION: REJECTED<br>'
            '<span style="font-size:0.85rem;font-weight:400;">Confidence: {:.1f}%</span></div>'.format(confidence*100),
            unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Applicant Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Credit Score", app_data["credit_score"])
    m2.metric("FOIR", "{:.1f}%".format(app_data["foir"] * 100))
    m3.metric("Loan-to-Income", "{:.2f}x".format(app_data["loan_to_annual_income_ratio"]))
    m4.metric("Past Default", "Yes" if app_data["past_default"] == 1 else "No")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Feature Contribution Breakdown")
    st.caption("How each feature moved the score toward Approval (+) or Rejection (-)")
    for item in cdf.to_dict("records"):
        cv = item["Contribution"]
        fn = item["Feature"]
        ac = abs(cv)
        bp = min(ac / 3.0 * 100, 100)
        if cv >= 0:
            bc = "feature-bar-positive"; ss = "+{:.4f}".format(cv); cl = "var(--accent-emerald)"; si = "\U0001F7E2"
        else:
            bc = "feature-bar-negative"; ss = "{:.4f}".format(cv); cl = "var(--accent-rose)"; si = "\U0001F534"
        dn = fn.replace("_", " ").replace("inr", "(INR)").title()
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
            '<span style="width:250px;font-size:0.85rem;color:var(--text-secondary);">'
            + si + ' ' + dn + '</span>'
            '<div style="flex:1;background:var(--bg-glass);border-radius:4px;height:8px;overflow:hidden;">'
            '<div class="' + bc + '" style="width:' + str(bp) + '%;"></div></div>'
            '<span style="width:80px;text-align:right;font-weight:600;font-size:0.85rem;color:'
            + cl + ';">' + ss + '</span></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Full Mathematical Breakdown", expanded=False):
        st.code(math_breakdown, language="text")
    st.info("Navigate to **Explainability Chat** to ask the AI about this decision.")

# ============================================================
# 6. SCREEN 3 - RAG EXPLAINABILITY CHATBOT
# ============================================================

elif screen == "Explainability Chat":
    st.markdown(
        '<div style="margin-bottom:1.5rem;">'
        '<p class="hero-title" style="font-size:1.5rem;">\U0001F4AC Explainability Chatbot</p>'
        '<p class="hero-subtitle">Ask questions about the AI decision \u2014 powered by bank policy & model math.</p>'
        '</div>', unsafe_allow_html=True)
    if "ai_decision" not in st.session_state:
        st.info("Please complete the **AI Assessment** first so there is a decision to explain.")
        st.stop()
    app_data = st.session_state["application"]
    decision = st.session_state["ai_decision"]
    math_breakdown = st.session_state["math_breakdown"]
    system_prompt = (
        "You are a helpful loan officer assistant for Bharat Horizon Bank. "
        "Your job is to explain the AI home loan decision to the applicant in clear, plain English.\n\n"
        "=== BANK POLICY DOCUMENT ===\n" + policy_text + "\n\n"
        "=== APPLICANT DATA ===\n"
        "- Monthly Income: INR " + "{:,}".format(app_data["monthly_income_inr"]) + "\n"
        "- Credit Score: " + str(app_data["credit_score"]) + "\n"
        "- Requested Loan Amount: INR " + "{:,}".format(app_data["loan_amount_inr"]) + "\n"
        "- Loan Tenure: " + str(app_data["loan_tenure_years"]) + " years\n"
        "- Existing EMI/Debts: INR " + "{:,}".format(app_data["existing_emi_inr"]) + "\n"
        "- Employment Years: " + str(app_data["employment_years"]) + "\n"
        "- Employment Type: " + app_data["employment_type"] + "\n"
        "- Past Default: " + ("Yes" if app_data["past_default"] == 1 else "No") + "\n"
        "- Computed FOIR: " + "{:.1f}%".format(app_data["foir"] * 100) + "\n"
        "- Loan-to-Annual-Income Ratio: " + "{:.2f}x".format(app_data["loan_to_annual_income_ratio"]) + "\n"
        "- Proposed EMI: INR " + "{:,.0f}".format(app_data["proposed_emi"]) + "\n\n"
        "=== AI DECISION & MATH ===\n"
        "Decision: " + decision + "\n" + math_breakdown + "\n\n"
        "=== INSTRUCTIONS ===\n"
        "1. Always reference specific policy rules (e.g., Rule 2.1).\n"
        "2. Map feature contributions to relevant policy rules.\n"
        "3. Be empathetic and professional. Give actionable advice.\n"
        "4. Never fabricate rules. Only use the policy document above.\n"
        "5. Keep responses concise but thorough."
    )
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_input = st.chat_input("Ask about the loan decision... (e.g., 'Why was I rejected?')")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        msgs = [{"role": "system", "content": system_prompt}]
        for msg in st.session_state["chat_history"]:
            msgs.append({"role": msg["role"], "content": msg["content"]})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_llm(msgs)
            st.markdown(response)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})

# ============================================================
# 7. SCREEN 4 - HUMAN-IN-THE-LOOP REVIEW
# ============================================================

elif screen == "Human Review":
    st.markdown(
        '<div style="margin-bottom:1.5rem;">'
        '<p class="hero-title" style="font-size:1.5rem;">\U0001F464 Human-in-the-Loop Review</p>'
        '<p class="hero-subtitle">Internal dashboard for the bank underwriter \u2014 final authority.</p>'
        '</div>', unsafe_allow_html=True)
    if "ai_decision" not in st.session_state:
        st.info("Please complete the **AI Assessment** first.")
        st.stop()
    app_data = st.session_state["application"]
    decision = st.session_state["ai_decision"]
    math_breakdown = st.session_state["math_breakdown"]
    if app_data.get("final_status") == "Closed":
        fd = app_data.get("final_decision", "N/A")
        if "Approv" in fd:
            st.markdown(
                '<div class="final-approved">'
                '<div style="font-size:2rem;margin-bottom:0.5rem;">\u2705</div>'
                '<div style="font-size:1.3rem;font-weight:700;color:var(--accent-emerald);">'
                'Application CLOSED \u2014 Final Decision: ' + fd + '</div>'
                '<div style="color:var(--text-secondary);margin-top:0.5rem;">'
                'Reviewed and finalized by a human underwriter.</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="final-rejected">'
                '<div style="font-size:2rem;margin-bottom:0.5rem;">\u274C</div>'
                '<div style="font-size:1.3rem;font-weight:700;color:var(--accent-rose);">'
                'Application CLOSED \u2014 Final Decision: ' + fd + '</div>'
                '<div style="color:var(--text-secondary);margin-top:0.5rem;">'
                'Reviewed and finalized by a human underwriter.</div></div>',
                unsafe_allow_html=True)
        st.stop()
    # Summary card
    def _f(v): return "{:,}".format(v)
    st.markdown(
        '<div class="summary-card">'
        '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
        'color:var(--text-muted);margin-bottom:1rem;">Applicant Summary</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">'
        '<div><div style="font-size:0.75rem;color:var(--text-muted);">Monthly Income</div>'
        '<div style="font-size:1.1rem;font-weight:600;">INR '+_f(app_data["monthly_income_inr"])+'</div></div>'
        '<div><div style="font-size:0.75rem;color:var(--text-muted);">Credit Score</div>'
        '<div style="font-size:1.1rem;font-weight:600;">'+str(app_data["credit_score"])+'</div></div>'
        '<div><div style="font-size:0.75rem;color:var(--text-muted);">Loan Amount</div>'
        '<div style="font-size:1.1rem;font-weight:600;">INR '+_f(app_data["loan_amount_inr"])+'</div></div>'
        '<div><div style="font-size:0.75rem;color:var(--text-muted);">Tenure</div>'
        '<div style="font-size:1.1rem;font-weight:600;">'+str(app_data["loan_tenure_years"])+' years</div></div>'
        '<div><div style="font-size:0.75rem;color:var(--text-muted);">Employment</div>'
        '<div style="font-size:1.1rem;font-weight:600;">'+app_data["employment_type"]+' ('+str(app_data["employment_years"])+' yrs)</div></div>'
        '<div><div style="font-size:0.75rem;color:var(--text-muted);">Past Default</div>'
        '<div style="font-size:1.1rem;font-weight:600;">'+("Yes" if app_data["past_default"]==1 else "No")+'</div></div>'
        '<div><div style="font-size:0.75rem;color:var(--text-muted);">FOIR</div>'
        '<div style="font-size:1.1rem;font-weight:600;">{:.1f}%</div></div>'.format(app_data["foir"]*100)+
        '<div><div style="font-size:0.75rem;color:var(--text-muted);">Loan-to-Income</div>'
        '<div style="font-size:1.1rem;font-weight:600;">{:.2f}x</div></div>'.format(app_data["loan_to_annual_income_ratio"])+
        '<div><div style="font-size:0.75rem;color:var(--text-muted);">Proposed EMI</div>'
        '<div style="font-size:1.1rem;font-weight:600;">INR '+_f(app_data["proposed_emi"])+'</div></div>'
        '</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    ai_col, summary_col = st.columns([1, 2], gap="large")
    with ai_col:
        st.markdown("##### AI Recommendation")
        if decision == "Approved":
            st.markdown('<div class="decision-approved" style="font-size:1.1rem;padding:1rem;">\u2705 APPROVED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="decision-rejected" style="font-size:1.1rem;padding:1rem;">\u274C REJECTED</div>', unsafe_allow_html=True)
    with summary_col:
        st.markdown("##### AI Summary (LLM-generated)")
        if "llm_summary" not in st.session_state:
            sp = (
                "Based on the following, write exactly 3 concise sentences on why the AI made this decision. "
                "Reference specific policy rules.\n\n"
                "Credit Score: "+str(app_data["credit_score"])+", "
                "Income: INR "+_f(app_data["monthly_income_inr"])+", "
                "Loan: INR "+_f(app_data["loan_amount_inr"])+", "
                "FOIR: {:.1f}%, ".format(app_data["foir"]*100)+
                "Loan-to-Income: {:.2f}x, ".format(app_data["loan_to_annual_income_ratio"])+
                "Default: "+("Yes" if app_data["past_default"]==1 else "No")+", "
                "Employment: "+app_data["employment_type"]+" "+str(app_data["employment_years"])+" yrs\n\n"
                "Decision: "+decision+"\n\nPOLICY:\n"+policy_text+"\n\nMATH:\n"+math_breakdown+
                "\n\nWrite ONLY the 3-sentence summary."
            )
            with st.spinner("Generating AI summary..."):
                summary = chat_with_llm([
                    {"role": "system", "content": "You are a concise bank analyst. Write exactly 3 sentences."},
                    {"role": "user", "content": sp}])
            st.session_state["llm_summary"] = summary
        st.markdown(
            '<div class="glass-card"><div style="font-size:0.9rem;line-height:1.7;color:var(--text-secondary);">'
            + st.session_state["llm_summary"] + '</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("View Model Math Breakdown"):
        st.code(math_breakdown, language="text")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Final Human Decision")
    st.caption("As the authorized underwriter, make the final call on this application.")
    b1, b2, _ = st.columns([1, 1, 2])
    with b1:
        if st.button("\u2705  Final Approve", use_container_width=True, type="primary"):
            st.session_state["application"]["final_decision"] = "Approved"
            st.session_state["application"]["final_status"] = "Closed"
            st.rerun()
    with b2:
        if st.button("\u274C  Final Reject", use_container_width=True):
            st.session_state["application"]["final_decision"] = "Rejected"
            st.session_state["application"]["final_status"] = "Closed"
            st.rerun()
