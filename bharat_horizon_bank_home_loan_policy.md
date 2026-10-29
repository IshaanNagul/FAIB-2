# Bharat Horizon Bank - Retail Home Loan Underwriting Policy

**Document Version:** 1.0
**Effective Date:** Current Year
**Scope:** Automated & Human-in-the-loop Retail Home Loan Assessment

## 1. Credit Score Guidelines
* **Rule 1.1 (Minimum Threshold):** The primary applicant must have a minimum credit score of 650. Applications with scores below 650 are subject to automatic rejection.
* **Rule 1.2 (Prime Consideration):** Applicants with a credit score of 750 or above are classified as "Prime" and are highly favored for approval, provided basic income criteria are met.

## 2. Fixed Obligation to Income Ratio (FOIR)
* **Rule 2.1 (Maximum FOIR):** An applicant's FOIR (including the proposed EMI) must not exceed 50% of their net monthly income. A FOIR > 50% is a primary driver for rejection.
* **Rule 2.2 (Optimal FOIR):** A FOIR of 40% or below is considered optimal and significantly increases the probability of an automated AI approval.

## 3. Loan-to-Annual-Income Constraints
* **Rule 3.1 (Maximum Borrowing Limit):** The total requested home loan amount must not exceed 5 times the applicant's gross annual income. 
* **Rule 3.2 (High-Risk Ratio):** If the Loan-to-Annual-Income ratio is between 4 and 5, the application requires a credit score of at least 700 to mitigate the higher risk profile.

## 4. Default History & Risk Profile
* **Rule 4.1 (Zero Tolerance for Recent Defaults):** Any history of loan or credit card default in the past 5 years will result in an immediate automated rejection, regardless of current income or employment status.
* **Rule 4.2 (Fraud Prevention):** Any discovered discrepancies between declared income and verified tax/bank records exceeding 5% will result in an automatic decline.

## 5. Employment & Income Stability
* **Rule 5.1 (Minimum Tenure):** The applicant must have a minimum of 2 years (24 months) of continuous employment history.
* **Rule 5.2 (Employment Type Verification):** Salaried employees must provide at least 6 months of continuous payslips. Self-employed individuals must provide a minimum of 2 years of audited financial statements.
* **Rule 5.3 (Income Drop):** A recent drop in annual income of more than 20% compared to the previous year requires the AI to flag the application for mandatory secondary human review.

## 6. General Eligibility & Co-Applicants
* **Rule 6.1 (Age Criteria):** The applicant must be at least 21 years old at the time of application and the loan must mature before the applicant reaches 65 years of age.
* **Rule 6.2 (Co-Applicant Benefit):** Adding a co-applicant (spouse or immediate family) allows their income to be combined, which can lower the effective FOIR and Loan-to-Annual-Income ratio.
* **Rule 6.3 (Missing Data):** Any application with incomplete mandatory fields (e.g., missing employment years or income details) will be placed in a "Hold" status and will not be processed by the ML model until rectified.

## 7. Human Override & Exceptions
* **Rule 7.1 (Human Authorization):** A human underwriter holds the final authority to override an AI-generated rejection if the applicant possesses liquid assets (e.g., fixed deposits, mutual funds) capable of covering at least 24 months of the proposed EMI.