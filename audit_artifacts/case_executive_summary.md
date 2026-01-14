# Pre-Deployment AI System Risk Audit: Executive Summary
## TransactionGuard AI Model Training Data
**Audit Date:** 2026-01-14 16:13 UTC
**Auditor:** Sarah Chen (Data Risk Lead)
**Dataset ID:** ccc68d0c-cc79-49e3-85b6-2be8f180909e
**Source System:** CoreBanking_ETL, **Ingestion Date:** 2023-01-15
**Label Column:** `is_fraud`
**Sensitive Attributes:** `customer_gender`

---
## 1. Overall Audit Status
- **DUPLICATES** on `Dataset`: Dataset contains 98 duplicate rows, which is 0.97% of total records. This exceeds the threshold of 0.10%. Potential for biased model training.

**Overall Audit Verdict:** **FAIL**

## 2. Key Risks Identified
### Data Quality Findings (FAIL)
  - Significant **DUPLICATE** rows (0.97%) potentially skewing training data.

## 3. Mitigation Recommendations
Based on the audit findings, the following mitigation actions are recommended:
- **Duplicate Rows:** De-duplicate the training dataset to ensure each transaction record is unique. Review data ingestion pipelines to prevent future duplicate entries.

---
## 4. Next Steps
- Present this report to the AI Governance Committee for review.
- Initiate work on identified mitigation strategies with the Data Engineering and ML teams.
- Re-audit the dataset post-mitigation to confirm effectiveness.
- Document findings and actions in GlobalTrust Financial's AI Risk Register.