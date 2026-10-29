# Bharat Horizon Bank - Home Loan Assessment System

A 4-screen Streamlit MVP for automated and explainable home loan underwriting, combining on-the-fly machine learning (Logistic Regression), bank policy reasoning via RAG (OpenRouter LLM), and human-in-the-loop decision-making.

> **Disclaimer**: Built for educational and college project demonstration purposes only.

---

## Architecture & Screens

1. **Loan Application Form** (`Screen 1`):
   - Captures applicant financial and credit parameters (Income, Loan Amount, Tenure, EMI, Credit Score, Employment).
   - Computes financial ratios dynamically: Proposed EMI, Fixed Obligation to Income Ratio (FOIR), and Loan-to-Annual-Income Ratio.
2. **AI Assessment** (`Screen 2`):
   - Logistic Regression trained on-the-fly (`@st.cache_resource`) using `bharat_horizon_home_loan_dataset_v2.xlsx`.
   - Feature contribution visual breakdown displaying exact weights and impact on loan approval log-odds.
   - Expandable complete mathematical formula transparency.
3. **Explainability Chatbot (RAG)** (`Screen 3`):
   - Conversational AI powered by OpenRouter (`google/gemini-2.5-flash`).
   - Grounded directly in `bharat_horizon_bank_home_loan_policy.md` and the applicant's exact mathematical profile.
4. **Human-in-the-Loop Review** (`Screen 4`):
   - Underwriter decision dashboard featuring a 3-sentence LLM policy review.
   - Human authority buttons (*Final Approve* / *Final Reject*) that lock the case and finalize the application status.

---

## Setup & Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/IshaanNagul/FAIB-2.git
cd FAIB-2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Secrets
Create `.streamlit/secrets.toml`:
```toml
OPENROUTER_API_KEY = "your-openrouter-api-key-here"
```

### 4. Run Streamlit
```bash
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
