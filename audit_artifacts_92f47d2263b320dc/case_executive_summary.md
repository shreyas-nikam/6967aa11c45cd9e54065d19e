# Pre-Deployment AI System Risk Audit: Executive Summary
## TransactionGuard AI Model Training Data

**Audit Date:** 2026-01-14 17:15 UTC

**Auditor:** Sarah Chen (Data Risk Lead)

**Dataset ID:** 87c6b810-e34c-470b-bcce-3dbbbab180c1

**Source System:** CoreBanking_ETL, **Ingestion Date:** 2023-01-15

**Label Column:** `is_fraud`

**Sensitive Attributes:** `customer_gender`, `customer_age_group`, `customer_region`

---
## 1. Overall Audit Status

### Bias Metrics Findings (FAIL)
- **DISPARATE_IMPACT** for `customer_age_group` group `18-30` (vs. `31-50`): Value 1.320. The positive outcome rate for '18-30' is 6.26% compared to 4.74% for '31-50'. DIR of 1.320 indicates a significant disparity in positive outcomes.

**Overall Audit Verdict:** **FAIL**

## 2. Key Risks Identified
  - Significant **BIAS** detected in `customer_age_group` for group `18-30` (Metric: DISPARATE_IMPACT). Potential for unfair outcomes.

## 3. Mitigation Recommendations
Based on the audit findings, the following mitigation actions are recommended:
- **Bias in Sensitive Attributes:**
  - For `customer_age_group` (Group: `18-30`), address the `DISPARATE_IMPACT` disparity. Consider data re-sampling (e.g., oversampling under-represented positive outcomes, or undersampling over-represented negative outcomes), re-weighting, or exploring fairness-aware learning algorithms.
  - Conduct a root-cause analysis to understand why these disparities exist in the data (e.g., historical data collection practices, inherent socio-economic factors) to inform long-term data strategy.

---
## 4. Next Steps
- Present this report to the AI Governance Committee for review.
- Initiate work on identified mitigation strategies with the Data Engineering and ML teams.
- Re-audit the dataset post-mitigation to confirm effectiveness.
- Document findings and actions in GlobalTrust Financial's AI Risk Register.