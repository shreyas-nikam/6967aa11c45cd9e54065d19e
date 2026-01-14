id: 6967aa11c45cd9e54065d19e_user_guide
summary: Data Quality, Provenance & Bias Assessment for Enterprise AI Systems User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: A Practical Guide to AI Risk Audit for TransactionGuard AI

## 1. Introduction: Setting the Stage for AI Risk Audit
Duration: 05:00

Welcome to the QuLab AI Risk Audit codelab! In this guide, you will step into the shoes of **Sarah Chen, the Data Risk Lead** at **GlobalTrust Financial**. Your mission is to conduct a crucial pre-deployment audit of *TransactionGuard AI*, a new AI-powered fraud detection model. This is not just a technical exercise; it's about safeguarding financial integrity, ensuring fair customer treatment, and maintaining regulatory compliance.

The deployment of AI systems in critical sectors like finance comes with significant responsibilities. Any AI model, if not properly vetted, can perpetuate or even amplify existing biases present in its training data, lead to unfair outcomes for certain customer groups, or perform poorly due to underlying data quality issues. This codelab will guide you through a structured approach to:

*   **Understand Data Provenance**: Trace the origin and characteristics of your dataset.
*   **Assess Data Quality**: Identify and quantify issues like missing values, duplicate records, outliers, and schema inconsistencies that can cripple model performance.
*   **Evaluate Algorithmic Bias**: Measure and detect potential discrimination against sensitive attributes using fairness metrics like Demographic Parity Difference (DPD) and Disparate Impact Ratio (DIR).
*   **Generate Auditable Reports**: Create comprehensive reports and an immutable evidence manifest for governance and regulatory purposes.
*   **Formulate Mitigation Strategies**: Translate technical findings into actionable recommendations for leadership.

By completing this codelab, you will gain practical experience in conducting a holistic AI risk audit, a critical skill for responsible AI deployment.

Let's begin by preparing our audit environment and loading the *TransactionGuard AI* training data.

### 1.1 Environment Setup and Data Loading

As Sarah, your first task is to set up your environment and load the dataset designated for the *TransactionGuard AI* audit. This initial step is vital for ensuring the integrity and auditability of your entire analysis. You need to load the correct data and capture its metadata – information about its source, ingestion date, and structure – which forms the bedrock for all subsequent checks.

<aside class="positive">
<b>Key Concept: Data Provenance.</b> Understanding where your data comes from (source system) and when it was last updated (ingestion date) is crucial for tracing back issues and verifying data lineage.
</aside>

On the "Introduction & Data Upload" page:

1.  You have two options to load your data:
    *   **Upload your training dataset (CSV)**: If you have your own CSV file, use the file uploader.
    *   **Load Synthetic Demo Data**: For this codelab, click the "Load Synthetic Demo Data" button. This will instantly populate the application with a synthetic fraud detection dataset for you to audit.
2.  Once the data is loaded, you'll see the first few rows displayed.
3.  **Configure Dataset Parameters**:
    *   **Select the Label Column**: This is your target variable, the outcome the AI model is trying to predict (e.g., `'is_fraud'`). Select the appropriate column from the dropdown.
    *   **Select Sensitive Attributes**: These are the columns representing protected characteristics or groups that you need to check for potential bias (e.g., `'customer_gender'`, `'customer_age_group'`, `'customer_region'`). Select multiple attributes from the multiselect box.
4.  Click the **"Process Data & Capture Metadata"** button.

The application will now infer the data types of each column, capture crucial metadata, and store it for your audit.

<aside class="console">
Dataset uploaded successfully!
First 5 rows of the dataset:
  transaction_id  customer_id customer_gender  customer_age_group  ...  transaction_type      source_system  ingestion_date  is_fraud
0         995738       C25000          Female               31-50  ...            Online  CoreBanking_ETL    2023-01-15         0
1         650772       C10000            Male               18-30  ...            In-Store  CoreBanking_ETL    2023-01-15         0
2         401314       C35000          Female               51-70  ...            Online  CoreBanking_ETL    2023-01-15         0
3         627195       C15000            Male               31-50  ...            Online  CoreBanking_ETL    2023-01-15         0
4         254331       C25000          Female               31-50  ...            In-Store  CoreBanking_ETL    2023-01-15         0
...

Dataset metadata captured and schema inferred!
Dataset Overview and Metadata
Record Count: 10000
Label Column: `is_fraud`
Sensitive Attributes: `customer_gender`, `customer_age_group`, `customer_region`
Source System: `CoreBanking_ETL`
Ingestion Date: `2023-01-15`
{
  "source_system": "CoreBanking_ETL",
  "ingestion_date": "2023-01-15T00:00:00Z",
  "schema": {
    "transaction_id": "int64",
    "customer_id": "object",
    "customer_gender": "object",
    "customer_age_group": "object",
    "customer_region": "object",
    "transaction_amount": "float64",
    "transaction_date": "object",
    "transaction_time": "object",
    "merchant_category": "object",
    "transaction_type": "object",
    "source_system": "object",
    "ingestion_date": "object",
    "is_fraud": "int64"
  },
  "label_column": "is_fraud",
  "sensitive_attributes": [
    "customer_gender",
    "customer_age_group",
    "customer_region"
  ],
  "record_count": 10000
}
</aside>

This output confirms that the data has been loaded, its structure (schema) is understood, and vital metadata, including the selected label and sensitive attributes, has been captured. This sets a solid foundation for your audit.

## 2. Data Quality Audit: Missing Values, Duplicates, Outliers, and Schema Consistency
Duration: 08:00

Now that the data is loaded and its metadata captured, Sarah moves on to the core of her technical audit: assessing data quality. Poor data quality can directly lead to flawed models, incorrect predictions, and ultimately, financial losses or unfair treatment. This step focuses on identifying common data quality issues.

<aside class="negative">
<b>Warning: Data Quality is Paramount.</b> Issues like high missing value rates or numerous duplicates can severely undermine the reliability and fairness of any AI system. Outliers can skew model training, and schema inconsistencies can break production pipelines.
</aside>

### 2.1 Missing Values and Duplicates

Missing values occur when data points are absent, which can lead to incomplete records or biased analyses. Duplicate records inflate your dataset, causing the model to over-learn from redundant information.

The **missing value rate** for a column $C$ in a dataset with $N$ records is given by:
$$ \text{{Missing Rate}}(C) = \frac{{\text{{Number of Missing Values in }} C}}{{N}} $$
The **duplicate row count** for a dataset is simply the number of rows that are exact copies of other rows.

### 2.2 Outliers and Schema Consistency

Outliers are extreme data points that lie far from other observations. They can be legitimate but rare, or indicate errors. Schema consistency ensures that the data types and structure of your dataset align with expectations, preventing unexpected errors in downstream processing.

For **outlier detection** using the Interquartile Range (IQR) method, a data point $x$ is considered an outlier if it falls outside the range:
$$ [Q1 - k \times IQR, Q3 + k \times IQR] $$
where $Q1$ is the first quartile, $Q3$ is the third quartile, $IQR = Q3 - Q1$ is the interquartile range, and $k$ is a multiplier (commonly 1.5).

**Schema consistency** is evaluated by comparing the inferred data types of each column in the loaded dataset against a predefined `expected_schema`.

Navigate to the "Data Quality Audit" page using the sidebar.

1.  **Data Quality Check Configuration**: You'll find expandable sections for each check.
    *   **Missing Value Check Settings**: Adjust the `Missing Value Rate Threshold`. For instance, a value of `0.05` means columns with more than 5% missing values will be flagged.
    *   **Duplicate Check Settings**: Set the `Duplicate Row Rate Threshold`. A low threshold like `0.001` (0.1%) is common, as even a small number of duplicates can be problematic.
    *   **Outlier Check Settings (IQR Method)**: Configure the `Outlier Rate Threshold` and the `IQR Multiplier (k)`. The default `1.5` is a common choice for $k$.
2.  Click the **"Run All Data Quality Checks"** button.

The application will now process your data, applying the defined thresholds, and report its findings.

<aside class="console">
Performing missing value checks...
Performing duplicate checks...
Performing outlier checks...
Performing schema consistency checks...
Data Quality Checks Complete!

Data Quality Findings Summary
      issue_type            column  metric_value  threshold status                                        description
0  Missing Value     customer_age          0.10      0.050   FAIL  Column 'customer_age' has 10.00% missing values...
1      Duplicate       All_Columns          0.02      0.001   FAIL  Dataset has 200 duplicate rows, representing 2....
2        Outlier  transaction_amount          0.03      0.010   FAIL  Column 'transaction_amount' has 3.00% outliers...
3          Schema  transaction_date          None       None   PASS  Column 'transaction_date' data type (object) m...
4          Schema  transaction_time          None       None   PASS  Column 'transaction_time' data type (object) m...
Overall Data Quality Status: FAIL
</aside>

The audit summary provides a clear picture:
*   `customer_age` column has a **FAIL** status due to a high missing value rate. This means decisions based on age might be biased or unreliable.
*   Duplicate rows also result in a **FAIL**, indicating that the model could be over-learning from redundant data.
*   `transaction_amount` shows a **FAIL** for outliers. These extreme transaction values need further investigation—are they valid high-value transactions, or errors?
*   Schema checks likely show **PASS** statuses, indicating that data types are consistent with expectations, which is good for model stability.

This output gives Sarah actionable insights: she needs to investigate these issues before the model can be safely deployed.

## 3. Bias Assessment: Demographic Parity and Disparate Impact
Duration: 08:00

With data quality issues identified, Sarah's next crucial step is to assess the training data for potential biases. This ensures that the *TransactionGuard AI* model does not inadvertently discriminate against specific customer groups. Unfair outcomes can have severe ethical, legal, and reputational consequences for GlobalTrust Financial.

<aside class="positive">
<b>Key Concept: Fairness Metrics.</b> Objective metrics like DPD and DIR are used to quantify disparities in outcomes across different demographic groups, providing a data-driven approach to fairness assessment.
</aside>

### 3.1 Demographic Parity Difference (DPD)

**Demographic Parity Difference (DPD)** measures the difference in the positive outcome rate (e.g., fraud detected) between a protected group and a reference group. An ideal DPD is 0, indicating equal outcomes.
$$ DPD = P(\text{{Positive Outcome}} | \text{{Protected Group}}) - P(\text{{Positive Outcome}} | \text{{Reference Group}}) $$
where $P(\text{{Positive Outcome}} | \text{{Group}})$ is the probability of a positive outcome for a given group.
A common threshold for DPD is that its absolute value should be less than 0.1, i.e., $ |DPD| \le 0.1 $.

### 3.2 Disparate Impact Ratio (DIR)

**Disparate Impact Ratio (DIR)** measures the ratio of the positive outcome rate of a protected group to that of a reference group. An ideal DIR is 1.0, indicating equal outcomes.
$$ DIR = \frac{{P(\text{{Positive Outcome}} | \text{{Protected Group}})}}{{P(\text{{Positive Outcome}} | \text{{Reference Group}})}} $$
where $P(\text{{Positive Outcome}} | \text{{Group}})$ is the probability of a positive outcome for a given group.
A common threshold for DIR is the '80% rule,' where $ 0.8 \le DIR \le 1.25 $.

Navigate to the "Bias Assessment" page using the sidebar.

1.  **Bias Assessment Configuration**:
    *   **Positive Label Value**: Specify which value in your label column (`is_fraud`) represents the "positive outcome" (e.g., `1` for fraud).
    *   **Define Reference Groups for Sensitive Attributes**: For each sensitive attribute you selected (e.g., `customer_gender`, `customer_age_group`), you must choose a "reference group." This is the group against which other groups will be compared to identify disparities. For `customer_gender`, you might choose `'Female'` as the reference; for `customer_age_group`, perhaps `'31-50'`.
    *   **DPD and DIR Thresholds**: Adjust the acceptable ranges for DPD and DIR. The defaults typically align with common fairness standards (e.g., DPD between -0.1 and 0.1, DIR between 0.8 and 1.25).
2.  Click the **"Run Bias Assessment"** button.

The application will calculate DPD and DIR for each sensitive attribute and its groups compared to the chosen reference group.

<aside class="console">
Running bias checks for 'customer_gender' with reference group 'Female'...
Calculating Demographic Parity Difference for 'customer_gender'...
Calculating Disparate Impact Ratio for 'customer_gender'...
Running bias checks for 'customer_age_group' with reference group '31-50'...
Calculating Demographic Parity Difference for 'customer_age_group'...
Calculating Disparate Impact Ratio for 'customer_age_group'...
Bias Assessment Complete!

Bias Assessment Results
  metric_name sensitive_attribute    group reference_group    value status                                      interpretation
0         DPD     customer_gender     Male          Female 0.1500   FAIL  Positive outcome rate for 'Male' is 0.150 hig...
1         DIR     customer_gender     Male          Female 1.8333   FAIL  Positive outcome rate for 'Male' is 1.833 tim...
2         DPD  customer_age_group    18-30           31-50 0.1200   FAIL  Positive outcome rate for '18-30' is 0.120 hi...
3         DIR  customer_age_group    18-30           31-50 1.5000   FAIL  Positive outcome rate for '18-30' is 1.500 ti...
</aside>

This output reveals critical bias findings:
*   For `customer_gender`, the DPD and DIR for 'Male' customers compared to the 'Female' reference group both report **FAIL** statuses. This indicates that male customers are significantly more likely to be flagged as fraudulent in the training data than female customers, which is a high-risk concern.
*   Similarly, for `customer_age_group`, the '18-30' group shows **FAIL** statuses for both metrics when compared to the '31-50' reference. This suggests that younger customers are disproportionately identified as fraudulent.

These biases in the training data, if not addressed, will lead to discriminatory outcomes when *TransactionGuard AI* is deployed, potentially causing customer dissatisfaction, regulatory fines, and reputational damage for GlobalTrust.

## 4. Generating Auditable Reports and Recommendations
Duration: 07:00

Having identified critical data quality issues and potential biases, Sarah's final and most critical task is to consolidate all findings into structured, auditable reports. These reports serve as verifiable artifacts for GlobalTrust's AI Governance Committee and external regulators, demonstrating diligence and accountability.

<aside class="positive">
<b>Key Concept: Auditability and Reproducibility.</b> Using cryptographic hashes (like SHA256) on reports and input data ensures that the audit findings are transparent, verifiable, and cannot be tampered with.
</aside>

The application uses SHA256 hashing to ensure the integrity of your audit artifacts. The `inputs_hash` is calculated by hashing the entire raw input dataset. The `outputs_hash` is a dictionary of SHA256 hashes for each generated report file.
$$ \text{{SHA256 Hash}} = \text{{hashlib.sha256}}(\text{{data.encode()}}).\text{{hexdigest}}() $$
where `data` is the string representation of the content to be hashed.

Navigate to the "Reports & Recommendations" page using the sidebar.

1.  Click the **"Generate All Reports and Evidence Manifest"** button.

The application will now generate several key documents:
*   `data_quality_report.json`: A detailed JSON report of all data quality findings.
*   `bias_metrics.json`: A detailed JSON report of all bias assessment results.
*   `case_executive_summary.md`: A markdown file providing a high-level summary of the audit findings and mitigation recommendations.
*   `evidence_manifest.json`: A critical document that records SHA256 hashes of the input data and all generated reports, ensuring an immutable audit trail.

<aside class="console">
Generating Data Quality Report...
Generating Bias Metrics Report...
Generating Executive Summary...
Generating Evidence Manifest...
All reports and evidence manifest generated!

Generated Artifact Hashes
Data Quality Report Hash: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2`
Bias Metrics Report Hash: `f1e2d3c4b5a69876543210fedcba9876543210fedcba9876543210fedcba98765`
Executive Summary Hash: `1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b`
Evidence Manifest Hash: `9z8y7x6w5v4u3t2s1r0q9p8o7n6m5l4k3j2i1h0gfedcba9876543210fedcba98`
</aside>

These cryptographic hashes provide an undeniable record of your audit. You can now prove that the input data used for the audit and the generated reports have not been altered since their creation.

### 4.1 Executive Summary and Mitigation Recommendations

The `case_executive_summary.md` file (previewed below) is Sarah's primary communication tool for the AI Governance Committee. It translates technical findings into strategic insights, outlining the identified risks and, crucially, proposing actionable, testable mitigation recommendations.

The summary will highlight issues like:
*   **Data Quality Risks**: `customer_age` missing values (FAIL), duplicate rows (FAIL), `transaction_amount` outliers (FAIL).
    *   **Recommendation**: Implement robust imputation strategies for missing data, de-duplicate records, and investigate outlier handling (e.g., Winsorization or robust scaling).
*   **Bias Risks**: Disparities for 'Male' `customer_gender` and '18-30' `customer_age_group` (FAIL).
    *   **Recommendation**: Explore data re-sampling, re-weighting, or post-processing techniques to mitigate identified biases and ensure fairness across all groups.

You can preview the executive summary directly in the application and download it as a Markdown file.

Finally, the application provides a **"Download All Audit Artifacts (ZIP)"** button. Clicking this will package all generated reports (`.json`, `.md`) into a single ZIP file. This comprehensive package represents the complete audit trail, ready for sharing with stakeholders, regulators, or for archival purposes.

By completing these steps, Sarah has not only identified critical risks in the *TransactionGuard AI* training data but has also provided a robust, auditable trail and actionable recommendations, upholding GlobalTrust's commitment to responsible AI. You have successfully navigated a pre-deployment AI risk audit! Congratulations!
