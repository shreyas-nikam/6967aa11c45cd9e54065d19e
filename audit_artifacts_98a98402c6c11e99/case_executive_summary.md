# Pre-Deployment AI System Risk Audit: Executive Summary
## TransactionGuard AI Model Training Data
**Audit Date:** 2026-01-14 16:53 UTC
**Auditor:** Sarah Chen (Data Risk Lead)
**Dataset ID:** 3d4e43fb-b503-4570-be72-bbce348e820f
**Source System:** CoreBanking_ETL, **Ingestion Date:** 2023-01-15
**Label Column:** `is_fraud`
**Sensitive Attributes:** `customer_gender`, `customer_age_group`, `customer_region`

---
## 1. Overall Audit Status
- **MISSINGNESS** on `customer_age`: Column 'customer_age' has 5.98% missing values. This exceeds the threshold of 5.00%. High risk of data incompleteness.
- **DUPLICATES** on `Dataset`: Dataset contains 98 duplicate rows, which is 0.97% of total records. This exceeds the threshold of 0.10%. Potential for biased model training.
- **OUTLIERS** on `amount`: Column 'amount' has 129 outliers (1.28%) based on IQR multiplier 1.5. Bounds: [-34.84, 235.51]. This exceeds the threshold of 1.00%. Potential for distorted model training.

**Overall Audit Verdict:** **FAIL**

## 2. Key Risks Identified
### Data Quality Findings (FAIL)
  - High **MISSINGNESS** in `customer_age` (5.98%) affecting data completeness and potential model bias.
  - Significant **DUPLICATE** rows (0.97%) potentially skewing training data.
  - High rate of **OUTLIERS** in `amount` (1.28%) could distort model learning for transaction values.

## 3. Mitigation Recommendations
Based on the audit findings, the following mitigation actions are recommended:
- **Missing Values:** Implement a robust imputation strategy (e.g., median imputation, K-nearest neighbors) for `customer_age` to address high missingness. Investigate upstream data collection processes for completeness.
- **Duplicate Rows:** De-duplicate the training dataset to ensure each transaction record is unique. Review data ingestion pipelines to prevent future duplicate entries.
- **Outliers in `amount`:** Investigate the nature of high-value `amount` outliers. Consider Winsorization or robust scaling techniques during feature engineering if they represent legitimate but rare events, or flag for data cleansing if they are errors.

---
## 4. Next Steps
- Present this report to the AI Governance Committee for review.
- Initiate work on identified mitigation strategies with the Data Engineering and ML teams.
- Re-audit the dataset post-mitigation to confirm effectiveness.
- Document findings and actions in GlobalTrust Financial's AI Risk Register.