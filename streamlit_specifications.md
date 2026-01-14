
# Streamlit Application Specification: AI System Pre-Deployment Risk Audit Dashboard

## 1. Application Overview

**Purpose of the application:**
The **AI System Pre-Deployment Risk Audit Dashboard** serves as a critical tool for AI Governance Committees and Data Risk Leads, like Sarah Chen at GlobalTrust Financial, to conduct comprehensive, auditable assessments of AI model training data. Its primary goal is to identify, quantify, and document potential risks related to data quality, fairness (bias), and data provenance before a high-impact AI system, such as a fraud detection model, is deployed. This proactive risk management ensures compliance, mitigates ethical concerns, and safeguards against financial and reputational damage.

**High-level story flow of the application:**
The application simulates Sarah Chen's workflow as she audits the *TransactionGuard AI* fraud detection model's training data.

1.  **Introduction & Data Loading:** Sarah begins by uploading the training dataset. The system automatically infers its schema and captures essential metadata for provenance (source system, ingestion date). She confirms the label column and specifies sensitive attributes for bias analysis.
2.  **Data Quality Audit:** Next, Sarah dives into data quality. She uses configurable thresholds to assess missing values, duplicate records, data outliers (e.g., in transaction amounts), and schema consistency. The dashboard provides a clear PASS/FAIL status for each check, highlighting areas of concern.
3.  **Bias Assessment:** With data quality reviewed, Sarah investigates potential biases. She selects various sensitive attributes (e.g., customer gender, age group) and calculates formal fairness metrics like Demographic Parity Difference and Disparate Impact Ratio. The application displays the results, compares them against configurable fairness thresholds, and provides interpretative guidance, identifying groups facing disparate outcomes.
4.  **Reports & Recommendations:** Finally, Sarah consolidates all findings. The application automatically generates detailed JSON reports for data quality and bias metrics, an executive summary in Markdown, and a cryptographically hashed `EvidenceManifest.json`. This manifest ensures the entire audit is traceable and tamper-proof. Sarah can then download all these auditable artifacts as a ZIP file, ready for presentation to GlobalTrust Financial's AI Governance Committee.

This structured workflow enables Sarah to efficiently perform a rigorous audit, translate complex technical findings into actionable insights, and ensure the responsible deployment of AI systems.

---

# 2. Code Requirements

```python
import streamlit as st
import pandas as pd
import os
import json
import zipfile
from io import BytesIO
from datetime import datetime, timezone

# Assume all Python code for logic and Pydantic models is available in source.py
from source import (
    DataQualityIssue, BiasMetric, DatasetMetadata, DataQualityFinding, BiasMetricResult,
    Artifact, EvidenceManifest,
    perform_missing_value_check, perform_duplicate_check, perform_outlier_check_iqr,
    perform_schema_check, calculate_demographic_parity_difference,
    calculate_disparate_impact_ratio, calculate_sha256_hash,
    generate_data_quality_report_json, generate_bias_metrics_report_json,
    generate_evidence_manifest_json, generate_executive_summary_md,
    generate_synthetic_fraud_data
)
```

### `st.session_state` Initialization, Update, and Read

`st.session_state` is used to maintain the application's state across user interactions and page navigations, simulating a multi-page experience.

**Initialization (at the beginning of `app.py`):**
```python
if 'page' not in st.session_state:
    st.session_state.page = "Introduction & Data Upload" # Default page

if 'df' not in st.session_state:
    st.session_state.df = None # Stores the uploaded/generated DataFrame

if 'baseline_df' not in st.session_state: # Optional for future drift analysis
    st.session_state.baseline_df = None

if 'dataset_metadata' not in st.session_state:
    st.session_state.dataset_metadata = None # Stores DatasetMetadata Pydantic object

if 'label_column' not in st.session_state:
    st.session_state.label_column = None # User-selected label column

if 'sensitive_attributes' not in st.session_state:
    st.session_state.sensitive_attributes = [] # User-selected sensitive attributes

if 'expected_schema' not in st.session_state:
    st.session_state.expected_schema = {} # Inferred schema from the uploaded DF

if 'dq_findings' not in st.session_state:
    st.session_state.dq_findings = [] # List of DataQualityFinding Pydantic objects

if 'bias_results' not in st.session_state:
    st.session_state.bias_results = [] # List of BiasMetricResult Pydantic objects

if 'dq_report_hash' not in st.session_state:
    st.session_state.dq_report_hash = None # SHA256 hash of the generated DQ report

if 'bias_report_hash' not in st.session_state:
    st.session_state.bias_report_hash = None # SHA256 hash of the generated Bias report

if 'executive_summary_hash' not in st.session_state:
    st.session_state.executive_summary_hash = None # SHA256 hash of the generated Executive Summary

if 'evidence_manifest_hash' not in st.session_state:
    st.session_state.evidence_manifest_hash = None # SHA256 hash of the generated Evidence Manifest

if 'output_dir' not in st.session_state:
    st.session_state.output_dir = "audit_artifacts" # Directory for saving reports

# Configurable Thresholds for Data Quality
if 'dq_threshold_missingness' not in st.session_state:
    st.session_state.dq_threshold_missingness = 0.05 # Default 5%

if 'dq_threshold_duplicates' not in st.session_state:
    st.session_state.dq_threshold_duplicates = 0.001 # Default 0.1%

if 'dq_threshold_outliers' not in st.session_state:
    st.session_state.dq_threshold_outliers = 0.01 # Default 1%

# Configurable Thresholds for Bias Metrics
if 'bias_dpd_threshold_min' not in st.session_state:
    st.session_state.bias_dpd_threshold_min = -0.1

if 'bias_dpd_threshold_max' not in st.session_state:
    st.session_state.bias_dpd_threshold_max = 0.1

if 'bias_dir_threshold_min' not in st.session_state:
    st.session_state.bias_dir_threshold_min = 0.8

if 'bias_dir_threshold_max' not in st.session_state:
    st.session_state.bias_dir_threshold_max = 1.25

if 'reference_groups' not in st.session_state:
    st.session_state.reference_groups = {} # Dict to store reference group per sensitive attribute
```

### Application Structure and Flow

The application will use a sidebar for navigation and global settings. The main content area will dynamically render based on the selected page.

```python
st.sidebar.title("AI Risk Audit Navigation")
st.session_state.page = st.sidebar.selectbox(
    "Go to",
    ["Introduction & Data Upload", "Data Quality Audit", "Bias Assessment", "Reports & Recommendations"]
)

# Main Title
st.title("Pre-Deployment AI System Risk Audit: TransactionGuard AI")
st.subheader("For Data Risk Lead, Sarah Chen (GlobalTrust Financial)")

# --- Page: Introduction & Data Upload ---
if st.session_state.page == "Introduction & Data Upload":
    st.markdown(f"")
    st.markdown(f"## Case Study Introduction")
    st.markdown(f"")
    st.markdown(f"As **Sarah Chen, the Data Risk Lead** at **GlobalTrust Financial**, your primary responsibility is to safeguard the integrity and fairness of the institution's AI systems. GlobalTrust is on the cusp of deploying a new AI-powered fraud detection model, *TransactionGuard AI*, a high-stakes system designed to protect customers and the institution from financial crime. However, before it can go live, it must undergo a rigorous pre-deployment audit.")
    st.markdown(f"")
    st.markdown(f"Your task is to conduct a comprehensive assessment of the model's training data. This audit is crucial for identifying potential data quality issues, uncovering problematic biases against sensitive customer groups, and verifying data provenance. Failing to identify these risks upfront could lead to severe consequences: unfair outcomes for customers, substantial regulatory fines, reputational damage, and erosion of trust.")
    st.markdown(f"")
    st.markdown(f"This application simulates your workflow as you systematically analyze the *TransactionGuard AI* training dataset, generating the necessary evidence and reports for the AI Governance Committee. Your findings will directly inform the decision to deploy the model or necessitate further remediation.")
    st.markdown(f"")
    st.markdown(f"---")
    st.markdown(f"")
    st.markdown(f"## 1. Environment Setup and Data Loading")
    st.markdown(f"")
    st.markdown(f"Sarah's first step is always to prepare her analytical environment and load the dataset earmarked for audit. This ensures she has the necessary tools and the correct data to begin her investigation. She needs to ensure her environment has all the required libraries and then load the specific training data for the *TransactionGuard AI* model. For auditability and reproducibility, the dataset's metadata, including its source and ingestion date, must also be captured.")
    st.markdown(f"")

    st.subheader("Upload Training Data or Load Demo Data")
    uploaded_file = st.file_uploader("Upload your training dataset (CSV)", type=["csv"])

    col1, col2 = st.columns(2)
    with col1:
        load_demo_data_button = st.button("Load Synthetic Demo Data")

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
        st.caption(f"First 5 rows of the dataset:")
        st.dataframe(st.session_state.df.head())
    elif load_demo_data_button:
        st.session_state.df = generate_synthetic_fraud_data(num_records=10000)
        st.success("Synthetic demo data loaded successfully!")
        st.caption(f"First 5 rows of the dataset:")
        st.dataframe(st.session_state.df.head())
    
    if st.session_state.df is not None:
        st.subheader("Configure Dataset Parameters")
        
        # Select label column
        all_columns = st.session_state.df.columns.tolist()
        default_label = 'is_fraud' if 'is_fraud' in all_columns else all_columns[0]
        st.session_state.label_column = st.selectbox(
            "Select the Label Column (e.g., 'is_fraud')",
            options=all_columns,
            index=all_columns.index(default_label) if default_label in all_columns else 0,
            key="label_col_selector"
        )
        
        # Select sensitive attributes
        default_sensitive = [
            'customer_gender', 
            'customer_age_group', 
            'customer_region'
        ]
        pre_selected_sensitive = [
            attr for attr in default_sensitive if attr in all_columns
        ]
        st.session_state.sensitive_attributes = st.multiselect(
            "Select Sensitive Attributes (e.g., 'customer_gender', 'customer_age_group')",
            options=[col for col in all_columns if col != st.session_state.label_column],
            default=pre_selected_sensitive,
            key="sensitive_attr_selector"
        )

        if st.button("Process Data & Capture Metadata"):
            with st.spinner("Capturing metadata and inferring schema..."):
                # Infer schema for DatasetMetadata
                inferred_schema = {col: str(dtype) for col, dtype in st.session_state.df.dtypes.items()}
                
                # Try to extract source_system and ingestion_date from first row, or use defaults
                source_system = st.session_state.df['source_system'].iloc[0] if 'source_system' in st.session_state.df.columns else "Unknown"
                ingestion_date_str = str(st.session_state.df['ingestion_date'].iloc[0]) if 'ingestion_date' in st.session_state.df.columns else datetime.now(timezone.utc).isoformat()
                
                # Attempt to parse ingestion date robustly
                try:
                    ingestion_date = pd.to_datetime(ingestion_date_str)
                except Exception:
                    st.warning(f"Could not parse 'ingestion_date' '{ingestion_date_str}'. Using current UTC datetime.")
                    ingestion_date = datetime.now(timezone.utc)
                
                # Create DatasetMetadata object
                st.session_state.dataset_metadata = DatasetMetadata(
                    source_system=source_system,
                    ingestion_date=ingestion_date,
                    schema=inferred_schema,
                    label_column=st.session_state.label_column,
                    sensitive_attributes=st.session_state.sensitive_attributes,
                    record_count=len(st.session_state.df)
                )
                st.session_state.expected_schema = inferred_schema # Store the inferred schema for schema check

            st.success("Dataset metadata captured and schema inferred!")
            st.subheader("Dataset Overview and Metadata")
            st.markdown(f"**Record Count:** {st.session_state.dataset_metadata.record_count}")
            st.markdown(f"**Label Column:** `{st.session_state.dataset_metadata.label_column}`")
            st.markdown(f"**Sensitive Attributes:** {', '.join([f'`{attr}`' for attr in st.session_state.dataset_metadata.sensitive_attributes])}")
            st.markdown(f"**Source System:** `{st.session_state.dataset_metadata.source_system}`")
            st.markdown(f"**Ingestion Date:** `{st.session_state.dataset_metadata.ingestion_date.strftime('%Y-%m-%d')}`")
            
            st.json(st.session_state.dataset_metadata.model_dump_json(indent=2))

            st.markdown(f"")
            st.markdown(f"#### Explanation of Execution: Data Loading")
            st.markdown(f"Sarah's first review confirms that the data has been loaded correctly, reflecting typical financial transaction data. The captured `DatasetMetadata` provides a vital snapshot of the data's structure and origin (`source_system`, `ingestion_date`), which is essential for auditability and understanding lineage risks. For instance, knowing the data comes from 'CoreBanking_ETL' and was ingested on '2023-01-15' allows her to trace back any fundamental issues to a specific batch or system. The auto-inferred schema is crucial for the upcoming schema consistency checks.")

    else:
        st.info("Please upload a CSV dataset or load demo data to proceed.")

# --- Page: Data Quality Audit ---
elif st.session_state.page == "Data Quality Audit":
    st.markdown(f"")
    st.markdown(f"## 2. Data Quality Audit: Missing Values and Duplicates")
    st.markdown(f"")
    st.markdown(f"Sarah begins her technical audit by checking for fundamental data quality issues. Missing values can lead to incomplete records, skewed statistics, or model errors if not handled properly. Duplicate records, if left unaddressed, can artificially inflate data volumes, bias model training, and lead to inaccurate performance metrics. Identifying these early ensures the model is trained on clean and reliable data.")
    st.markdown(f"")
    st.markdown(f"### 2.1 Story + Context + Real-World Relevance")
    st.markdown(f"Missing values in critical features can degrade model performance and lead to unfair decisions. For instance, if `customer_age` is frequently missing for certain demographic groups, the fraud detection model might perform poorly for those groups. Duplicate transactions could falsely represent activity, making certain fraud patterns appear more common than they are. Sarah must quantify these issues to understand their potential impact.")
    st.markdown(f"")
    st.markdown(r"The **missing value rate** for a column $C$ in a dataset with $N$ records is given by:")
    st.markdown(r"$$ \text{{Missing Rate}}(C) = \frac{{\text{{Number of Missing Values in }} C}}{{N}} $$")
    st.markdown(r"where $C$ is a column and $N$ is the total number of records in the dataset.")
    st.markdown(f"")
    st.markdown(r"The **duplicate row count** for a dataset is simply the number of rows that are exact copies of other rows.")
    st.markdown(f"")

    if st.session_state.df is None:
        st.warning("Please upload or load data in the 'Introduction & Data Upload' section first.")
    else:
        st.subheader("Data Quality Check Configuration")
        with st.expander("Missing Value Check Settings"):
            st.session_state.dq_threshold_missingness = st.number_input(
                "Missing Value Rate Threshold (e.g., 0.05 for 5%)",
                min_value=0.0, max_value=1.0, value=st.session_state.dq_threshold_missingness, step=0.01,
                format="%.3f", key="mv_thresh"
            )
        with st.expander("Duplicate Check Settings"):
            st.session_state.dq_threshold_duplicates = st.number_input(
                "Duplicate Row Rate Threshold (e.g., 0.001 for 0.1%)",
                min_value=0.0, max_value=1.0, value=st.session_state.dq_threshold_duplicates, step=0.001,
                format="%.4f", key="dup_thresh"
            )

        st.markdown(f"")
        st.markdown(f"## 3. Data Quality Audit: Outliers and Schema Consistency")
        st.markdown(f"")
        st.markdown(f"Continuing her data quality audit, Sarah shifts her focus to identifying outliers and ensuring schema consistency. Outliers, extreme values in numerical features, can severely distort statistical analyses and model training, leading to inaccurate predictions or unstable models. Schema consistency ensures that the data types and structure align with expectations, preventing unexpected errors in downstream processing pipelines and model serving.")
        st.markdown(f"")
        st.markdown(f"### 3.1 Story + Context + Real-World Relevance")
        st.markdown(f"Outliers in transaction `amount` could represent legitimate high-value transactions, but they could also indicate data entry errors or even sophisticated fraud attempts. Incorrectly handling them can mislead the fraud model. Similarly, if a column expected to be numerical suddenly contains text (a schema drift), the model pipeline would break. Sarah needs to detect these issues proactively.")
        st.markdown(f"")
        st.markdown(r"For **outlier detection** using the Interquartile Range (IQR) method, a data point $x$ is considered an outlier if it falls outside the range:")
        st.markdown(r"$$ [Q1 - k \times IQR, Q3 + k \times IQR] $$")
        st.markdown(r"where $Q1$ is the first quartile, $Q3$ is the third quartile, $IQR = Q3 - Q1$ is the interquartile range, and $k$ is a multiplier (commonly 1.5).")
        st.markdown(f"")
        st.markdown(r"**Schema consistency** is evaluated by comparing the inferred data types of each column in the loaded dataset against a predefined `expected_schema`.")
        st.markdown(f"")

        with st.expander("Outlier Check Settings (IQR Method)"):
            st.session_state.dq_threshold_outliers = st.number_input(
                "Outlier Rate Threshold (e.g., 0.01 for 1%)",
                min_value=0.0, max_value=1.0, value=st.session_state.dq_threshold_outliers, step=0.001,
                format="%.3f", key="outlier_thresh"
            )
            # A common IQR multiplier is 1.5, allow modification
            iqr_multiplier = st.number_input(
                "IQR Multiplier (k in formula, e.g., 1.5)",
                min_value=1.0, max_value=3.0, value=1.5, step=0.1, key="iqr_multiplier"
            )

        if st.button("Run All Data Quality Checks"):
            st.session_state.dq_findings = [] # Reset findings

            with st.spinner("Performing missing value checks..."):
                st.session_state.dq_findings.extend(
                    perform_missing_value_check(st.session_state.df, threshold=st.session_state.dq_threshold_missingness)
                )
            with st.spinner("Performing duplicate checks..."):
                st.session_state.dq_findings.append(
                    perform_duplicate_check(st.session_state.df)
                )
            with st.spinner("Performing outlier checks..."):
                # Only check numerical columns for outliers
                numerical_cols = st.session_state.df.select_dtypes(include=np.number).columns
                for col in numerical_cols:
                    if col not in ['transaction_id', st.session_state.label_column]: # Skip IDs and label
                        st.session_state.dq_findings.append(
                            perform_outlier_check_iqr(st.session_state.df, col, threshold_iqr_multiplier=iqr_multiplier)
                        )
            with st.spinner("Performing schema consistency checks..."):
                if st.session_state.expected_schema:
                    st.session_state.dq_findings.extend(
                        perform_schema_check(st.session_state.df, st.session_state.expected_schema)
                    )
                else:
                    st.warning("Expected schema not defined. Please process data in 'Introduction & Data Upload' first.")

            st.success("Data Quality Checks Complete!")

        if st.session_state.dq_findings:
            st.subheader("Data Quality Findings Summary")
            dq_df = pd.DataFrame([f.model_dump() for f in st.session_state.dq_findings])
            dq_df_display = dq_df[['issue_type', 'column', 'metric_value', 'threshold', 'status', 'description']]
            st.dataframe(dq_df_display)

            overall_dq_status = "PASS" if all(f.status == "PASS" for f in st.session_state.dq_findings if f.status != "N/A") else "FAIL"
            st.metric(label="Overall Data Quality Status", value=overall_dq_status)

            st.markdown(f"")
            st.markdown(f"#### Explanation of Execution: Data Quality Checks")
            st.markdown(f"The audit reveals that the `customer_age` column has a significant missing value rate (likely exceeding GlobalTrust's 5% threshold), resulting in a **FAIL** status. Sarah knows this means imputations or advanced missing value handling strategies are required before training, as simply dropping these rows could remove valuable data or introduce further bias.")
            st.markdown(f"")
            st.markdown(f"Furthermore, the check identified duplicate rows, also exceeding the low duplicate threshold. This finding is a **FAIL** and signals that the dataset needs de-duplication. Ignoring duplicates would lead to the model over-learning from redundant data points, potentially skewing feature importance and performance metrics.")
            st.markdown(f"")
            st.markdown(f"The outlier check for the `amount` column likely reports a **FAIL** status. Sarah notes that these outliers, representing unusually high transaction values, could either be legitimate but rare events, or indicators of sophisticated fraudulent activity or data errors. This requires further investigation. Depending on their nature, Winsorization or robust scaling might be necessary.")
            st.markdown(f"")
            st.markdown(f"The schema consistency check should report mostly **PASS** statuses if the data types match the inferred schema. If there were any mismatches, it would flag a **FAIL**. Sarah understands that schema drift is a critical issue that can silently break production models. The results confirm that the data types are as expected, reassuring her that the data will fit the model's input requirements, or highlight any deviations.")

# --- Page: Bias Assessment ---
elif st.session_state.page == "Bias Assessment":
    st.markdown(f"")
    st.markdown(f"## 4. Bias Assessment: Demographic Parity and Disparate Impact")
    st.markdown(f"")
    st.markdown(f"A core part of Sarah's role as a Data Risk Lead is to ensure that GlobalTrust Financial's AI systems do not perpetuate or amplify existing societal biases. Unfair outcomes can lead to regulatory scrutiny, legal challenges, and severe reputational damage. She must quantify potential biases in the *TransactionGuard AI* training data, particularly concerning sensitive attributes like `customer_gender`, `customer_age_group`, and `customer_region`.")
    st.markdown(f"")
    st.markdown(f"### 4.1 Story + Context + Real-World Relevance")
    st.markdown(f"If the fraud detection model disproportionately flags transactions from specific demographic groups as fraudulent (even when the underlying behavior isn't inherently riskier), it indicates bias in the training data. This could lead to legitimate customers from these groups facing unnecessary transaction declines or additional scrutiny, creating a discriminatory experience. Sarah uses formal fairness metrics to objectively measure these disparities.")
    st.markdown(f"")
    st.markdown(r"**Demographic Parity Difference (DPD)** measures the difference in the positive outcome rate (e.g., fraud detected) between a protected group and a reference group. An ideal DPD is 0, indicating equal outcomes.")
    st.markdown(r"$$ DPD = P(\text{{Positive Outcome}} | \text{{Protected Group}}) - P(\text{{Positive Outcome}} | \text{{Reference Group}}) $$")
    st.markdown(r"where $P(\text{{Positive Outcome}} | \text{{Group}})$ is the probability of a positive outcome for a given group.")
    st.markdown(r"A common threshold for DPD is that its absolute value should be less than 0.1, i.e., $ |DPD| \le 0.1 $.")
    st.markdown(f"")
    st.markdown(r"**Disparate Impact Ratio (DIR)** measures the ratio of the positive outcome rate of a protected group to that of a reference group. An ideal DIR is 1.0, indicating equal outcomes.")
    st.markdown(r"$$ DIR = \frac{{P(\text{{Positive Outcome}} | \text{{Protected Group}})}}{{P(\text{{Positive Outcome}} | \text{{Reference Group}})}} $$")
    st.markdown(r"where $P(\text{{Positive Outcome}} | \text{{Group}})$ is the probability of a positive outcome for a given group.")
    st.markdown(r"A common threshold for DIR is the '80% rule,' where $ 0.8 \le DIR \le 1.25 $.")
    st.markdown(f"")

    if st.session_state.df is None or not st.session_state.sensitive_attributes or st.session_state.label_column is None:
        st.warning("Please upload data and define Label Column & Sensitive Attributes in the 'Introduction & Data Upload' section first.")
    else:
        st.subheader("Bias Assessment Configuration")
        POSITIVE_LABEL_VALUE = st.radio(
            "What is the positive label value for your target column (e.g., 1 for fraud)?",
            options=st.session_state.df[st.session_state.label_column].unique().tolist(),
            key="positive_label_val"
        )

        st.markdown(f"#### Define Reference Groups for Sensitive Attributes")
        for attr in st.session_state.sensitive_attributes:
            unique_groups = st.session_state.df[attr].dropna().astype(str).unique().tolist()
            if unique_groups:
                default_ref = st.session_state.reference_groups.get(attr)
                if default_ref not in unique_groups: # Ensure default is still valid
                    default_ref = unique_groups[0]
                st.session_state.reference_groups[attr] = st.selectbox(
                    f"Select Reference Group for '{attr}'",
                    options=unique_groups,
                    index=unique_groups.index(default_ref) if default_ref in unique_groups else 0,
                    key=f"ref_group_{attr}"
                )
            else:
                st.warning(f"No unique groups found for sensitive attribute '{attr}'. Skipping reference group selection.")

        with st.expander("Demographic Parity Difference (DPD) Thresholds"):
            st.session_state.bias_dpd_threshold_min = st.number_input(
                "DPD Minimum Threshold (e.g., -0.1)",
                min_value=-1.0, max_value=0.0, value=st.session_state.bias_dpd_threshold_min, step=0.01,
                format="%.2f", key="dpd_min_thresh"
            )
            st.session_state.bias_dpd_threshold_max = st.number_input(
                "DPD Maximum Threshold (e.g., 0.1)",
                min_value=0.0, max_value=1.0, value=st.session_state.bias_dpd_threshold_max, step=0.01,
                format="%.2f", key="dpd_max_thresh"
            )

        with st.expander("Disparate Impact Ratio (DIR) Thresholds"):
            st.session_state.bias_dir_threshold_min = st.number_input(
                "DIR Minimum Threshold (e.g., 0.8)",
                min_value=0.0, max_value=1.0, value=st.session_state.bias_dir_threshold_min, step=0.05,
                format="%.2f", key="dir_min_thresh"
            )
            st.session_state.bias_dir_threshold_max = st.number_input(
                "DIR Maximum Threshold (e.g., 1.25)",
                min_value=1.0, max_value=5.0, value=st.session_state.bias_dir_threshold_max, step=0.05,
                format="%.2f", key="dir_max_thresh"
            )

        if st.button("Run Bias Assessment"):
            st.session_state.bias_results = [] # Reset results
            
            for attr in st.session_state.sensitive_attributes:
                if attr in st.session_state.reference_groups:
                    ref_group = st.session_state.reference_groups[attr]
                    st.markdown(f"Running bias checks for **'{attr}'** with reference group **'{ref_group}'**...")
                    with st.spinner(f"Calculating Demographic Parity Difference for '{attr}'..."):
                        dpd_results = calculate_demographic_parity_difference(
                            st.session_state.df, attr, st.session_state.label_column,
                            POSITIVE_LABEL_VALUE, ref_group,
                            threshold_min=st.session_state.bias_dpd_threshold_min,
                            threshold_max=st.session_state.bias_dpd_threshold_max
                        )
                        st.session_state.bias_results.extend(dpd_results)
                    
                    with st.spinner(f"Calculating Disparate Impact Ratio for '{attr}'..."):
                        dir_results = calculate_disparate_impact_ratio(
                            st.session_state.df, attr, st.session_state.label_column,
                            POSITIVE_LABEL_VALUE, ref_group,
                            threshold_min=st.session_state.bias_dir_threshold_min,
                            threshold_max=st.session_state.bias_dir_threshold_max
                        )
                        st.session_state.bias_results.extend(dir_results)
                else:
                    st.warning(f"No reference group selected for sensitive attribute '{attr}'. Skipping bias calculation.")

            st.success("Bias Assessment Complete!")
        
        if st.session_state.bias_results:
            st.subheader("Bias Assessment Results")
            bias_df = pd.DataFrame([b.model_dump() for b in st.session_state.bias_results])
            bias_df_display = bias_df[['metric_name', 'sensitive_attribute', 'group', 'reference_group', 'value', 'status', 'interpretation']]
            st.dataframe(bias_df_display)

            st.markdown(f"")
            st.markdown(f"#### Explanation of Execution: Bias Assessment")
            st.markdown(f"The bias assessment reveals critical findings that Sarah needs to address. For `customer_gender`, the **Demographic Parity Difference** and **Disparate Impact Ratio** for 'Male' customers compared to the 'Female' reference group likely report **FAIL** statuses. This indicates that male customers are disproportionately flagged as fraudulent in the training data. This finding is a high-risk concern for GlobalTrust, potentially leading to unfair treatment and violating anti-discrimination regulations.")
            st.markdown(f"")
            st.markdown(f"Similarly, the `customer_age_group` assessment might show a **FAIL** for the '18-30' group compared to the '31-50' reference, indicating an elevated fraud rate in the training data for younger customers. Sarah recognizes that these biases in the training data, if not mitigated, will be learned and amplified by the *TransactionGuard AI* model, leading to discriminatory outcomes. This necessitates further investigation into the source of this disparity and potential data re-sampling or re-weighting strategies to promote fairness.")

# --- Page: Reports & Recommendations ---
elif st.session_state.page == "Reports & Recommendations":
    st.markdown(f"")
    st.markdown(f"## 5. Generating Auditable Reports and Evidence Manifest")
    st.markdown(f"")
    st.markdown(f"Having performed the detailed quality and bias checks, Sarah's next crucial step is to consolidate all findings into structured, auditable reports. This is not just about summarizing data; it's about creating verifiable artifacts that can be presented to internal governance committees and external regulators.")
    st.markdown(f"")
    st.markdown(f"### 5.1 Story + Context + Real-World Relevance")
    st.markdown(f"For GlobalTrust Financial, every AI system deployment requires a clear audit trail. Sarah's reports serve as the official record of the pre-deployment audit. The `data_quality_report.json` and `bias_metrics.json` provide granular details, while the `evidence_manifest.json` acts as a cryptographic fingerprint of the entire audit, linking findings to the specific input data and tools used. This level of detail is paramount for demonstrating compliance and accountability to stakeholders and regulators.")
    st.markdown(f"")
    st.markdown(r"The `inputs_hash` is calculated by hashing the entire raw input dataset. The `outputs_hash` is a dictionary of SHA256 hashes for each generated report file.")
    st.markdown(r"$$ \text{{SHA256 Hash}} = \text{{hashlib.sha256}}(\text{{data.encode()}}).\text{{hexdigest}}() $$")
    st.markdown(r"where `data` is the string representation of the content to be hashed.")
    st.markdown(f"")

    if st.session_state.df is None or not st.session_state.dq_findings or not st.session_state.bias_results or st.session_state.dataset_metadata is None:
        st.warning("Please complete 'Introduction & Data Upload', 'Data Quality Audit', and 'Bias Assessment' sections first.")
    else:
        # Ensure output directory exists
        os.makedirs(st.session_state.output_dir, exist_ok=True)

        DATA_QUALITY_REPORT_FILENAME = "data_quality_report.json"
        BIAS_METRICS_REPORT_FILENAME = "bias_metrics.json"
        EXECUTIVE_SUMMARY_FILENAME = "case_executive_summary.md"
        EVIDENCE_MANIFEST_FILENAME = "evidence_manifest.json"

        if st.button("Generate All Reports and Evidence Manifest"):
            with st.spinner("Generating Data Quality Report..."):
                st.session_state.dq_report_hash = generate_data_quality_report_json(
                    st.session_state.dq_findings,
                    os.path.join(st.session_state.output_dir, DATA_QUALITY_REPORT_FILENAME)
                )
            with st.spinner("Generating Bias Metrics Report..."):
                st.session_state.bias_report_hash = generate_bias_metrics_report_json(
                    st.session_state.bias_results,
                    os.path.join(st.session_state.output_dir, BIAS_METRICS_REPORT_FILENAME)
                )
            with st.spinner("Generating Executive Summary..."):
                st.session_state.executive_summary_hash = generate_executive_summary_md(
                    dataset_metadata=st.session_state.dataset_metadata,
                    dq_findings=st.session_state.dq_findings,
                    bias_results=st.session_state.bias_results,
                    output_path=os.path.join(st.session_state.output_dir, EXECUTIVE_SUMMARY_FILENAME)
                )
            
            with st.spinner("Generating Evidence Manifest..."):
                # Initial manifest creation
                st.session_state.evidence_manifest_hash = generate_evidence_manifest_json(
                    df=st.session_state.df,
                    dq_findings_hash=st.session_state.dq_report_hash,
                    bias_results_hash=st.session_state.bias_report_hash,
                    dq_report_filename=DATA_QUALITY_REPORT_FILENAME,
                    bias_report_filename=BIAS_METRICS_REPORT_FILENAME,
                    output_path=os.path.join(st.session_state.output_dir, EVIDENCE_MANIFEST_FILENAME)
                )
                
                # Manually update manifest with executive summary (as it's a separate step in the source)
                evidence_manifest_path = os.path.join(st.session_state.output_dir, EVIDENCE_MANIFEST_FILENAME)
                with open(evidence_manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                
                # Add executive summary artifact
                if EXECUTIVE_SUMMARY_FILENAME not in manifest_data['outputs_hash']:
                    manifest_data['outputs_hash'][EXECUTIVE_SUMMARY_FILENAME] = st.session_state.executive_summary_hash
                    manifest_data['artifacts'].append(
                        Artifact(file_name=EXECUTIVE_SUMMARY_FILENAME, 
                                 file_hash=st.session_state.executive_summary_hash, 
                                 description="Consolidated executive summary of audit findings and recommendations.").model_dump()
                    )
                
                # Recalculate manifest hash after updating
                manifest_content_updated = json.dumps(manifest_data, indent=4)
                with open(evidence_manifest_path, 'w') as f:
                    f.write(manifest_content_updated)
                st.session_state.evidence_manifest_hash = calculate_sha256_hash(manifest_content_updated)


            st.success("All reports and evidence manifest generated!")

        if st.session_state.evidence_manifest_hash:
            st.subheader("Generated Artifact Hashes")
            st.markdown(f"**Data Quality Report Hash:** `{st.session_state.dq_report_hash}`")
            st.markdown(f"**Bias Metrics Report Hash:** `{st.session_state.bias_report_hash}`")
            st.markdown(f"**Executive Summary Hash:** `{st.session_state.executive_summary_hash}`")
            st.markdown(f"**Evidence Manifest Hash:** `{st.session_state.evidence_manifest_hash}`")

            st.markdown(f"")
            st.markdown(f"#### Explanation of Execution: Reports and Evidence Manifest")
            st.markdown(f"Sarah now has three critical files in her `audit_artifacts` directory: `data_quality_report.json`, `bias_metrics.json`, and `evidence_manifest.json`. The JSON reports provide detailed, machine-readable summaries of all individual findings, including metric values, thresholds, and PASS/FAIL statuses. This granular detail is invaluable for developers and data engineers who will implement the mitigation strategies.")
            st.markdown(f"")
            st.markdown(f"Crucially, the `evidence_manifest.json` binds the entire audit together. It contains hashes of the input training data and each generated report. This cryptographic linkage ensures that:")
            st.markdown(f"1. **Provenance**: Any future review can verify exactly which dataset was audited.")
            st.markdown(f"2. **Integrity**: It can be proven that the reports have not been tampered with since their generation.")
            st.markdown(f"")
            st.markdown(f"For Sarah, this output is the backbone of her audit trail, allowing GlobalTrust to demonstrate robust governance and compliance with regulatory requirements, providing confidence in the audit process itself.")
            st.markdown(f"")

            st.markdown(f"---")
            st.markdown(f"")
            st.markdown(f"## 6. Executive Summary and Mitigation Recommendations")
            st.markdown(f"")
            st.markdown(f"The final and most critical step for Sarah is to synthesize all audit findings into a concise executive summary for the AI Governance Committee. This document must clearly articulate the key risks identified, provide a summary PASS/FAIL status, and propose actionable, testable mitigation recommendations. This is where technical findings are translated into strategic insights for leadership.")
            st.markdown(f"")
            st.markdown(f"### 6.1 Story + Context + Real-World Relevance")
            st.markdown(f"Sarah understands that the AI Governance Committee needs a high-level overview, not raw data. Her executive summary must highlight the most significant data quality and bias risks, explaining their potential impact on GlobalTrust and its customers. More importantly, she must provide concrete recommendations that her team or data engineering can implement to resolve these issues before the *TransactionGuard AI* model goes live. This direct connection from problem to solution demonstrates proactive risk management.")
            st.markdown(f"")
            
            # Display Executive Summary content
            executive_summary_path = os.path.join(st.session_state.output_dir, EXECUTIVE_SUMMARY_FILENAME)
            if os.path.exists(executive_summary_path):
                st.subheader("Executive Summary (Preview)")
                with open(executive_summary_path, 'r') as f:
                    executive_summary_content = f.read()
                st.markdown(executive_summary_content)
                st.download_button(
                    label="Download Executive Summary (Markdown)",
                    data=executive_summary_content,
                    file_name=EXECUTIVE_SUMMARY_FILENAME,
                    mime="text/markdown"
                )
            
            st.markdown(f"")
            st.markdown(f"#### Explanation of Execution: Executive Summary")
            st.markdown(f"The `case_executive_summary.md` file now provides a clear, high-level overview of the audit. For Sarah, this document is her primary communication tool with the AI Governance Committee. It summarizes the **FAIL** statuses for missing values, duplicates, outliers, and most critically, the detected bias in `customer_gender` and `customer_age_group`. Each identified risk is accompanied by concrete, testable mitigation recommendations, such as robust imputation for missing `customer_age` and re-sampling strategies to address the identified biases.")
            st.markdown(f"")
            st.markdown(f"By providing these actionable recommendations, Sarah is not just identifying problems; she is providing a pathway for remediation. This empowers the Committee to make an informed decision on the *TransactionGuard AI* model's deployment, knowing that a thorough audit has been conducted and a clear plan is in place to address the identified risks, upholding GlobalTrust's commitment to responsible AI. The updated `evidence_manifest.json` ensures that even this final report is fully auditable.")
            st.markdown(f"")

            st.subheader("Download All Audit Artifacts")
            # Create a BytesIO object to hold the zip file in memory
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(st.session_state.output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zf.write(file_path, arcname=os.path.relpath(file_path, st.session_state.output_dir))
            
            # Reset buffer position to the beginning
            zip_buffer.seek(0)

            st.download_button(
                label="Download All Audit Artifacts (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="audit_artifacts.zip",
                mime="application/zip",
                help="Includes data_quality_report.json, bias_metrics.json, case_executive_summary.md, and evidence_manifest.json"
            )

```
