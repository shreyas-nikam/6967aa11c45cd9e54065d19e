id: 6967aa11c45cd9e54065d19e_documentation
summary: Data Quality, Provenance & Bias Assessment for Enterprise AI Systems Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Data Quality, Provenance & Bias Assessment for Enterprise AI Systems Codelab

## Introduction to QuLab and AI Risk Auditing
Duration: 0:05

<aside class="positive">
Welcome to the QuLab codelab! This guide will walk you through building and understanding a Streamlit application designed for crucial pre-deployment AI system risk audits. The application simulates the workflow of a Data Risk Lead, **Sarah Chen** from GlobalTrust Financial, as she assesses the *TransactionGuard AI* model's training data.
</aside>

### The Critical Need for AI Risk Auditing

In today's regulatory landscape, ensuring the fairness, transparency, and reliability of AI systems is paramount, especially in high-stakes domains like finance. AI models, if trained on poor quality or biased data, can perpetuate discrimination, lead to financial losses, incur regulatory fines, and severely damage an organization's reputation.

This codelab will explain how QuLab addresses these challenges by providing a structured framework to:
*   **Assess Data Quality**: Identify issues like missing values, duplicates, and outliers that can degrade model performance.
*   **Evaluate Data Provenance**: Track the origin and lineage of data for auditability.
*   **Quantify Algorithmic Bias**: Measure disparities in model outcomes across sensitive demographic groups.
*   **Generate Auditable Artifacts**: Produce comprehensive reports and an evidence manifest for governance and compliance.

The core concepts we'll explore include:
*   **Data Quality Metrics**: Missing value rates, duplicate counts, outlier detection (IQR method).
*   **Fairness Metrics**: Demographic Parity Difference (DPD) and Disparate Impact Ratio (DIR).
*   **Data Provenance**: Capturing metadata like source systems and ingestion dates.
*   **Audit Trails**: Using cryptographic hashes to ensure report integrity and traceability.

By the end of this codelab, you will have a comprehensive understanding of the QuLab application's functionalities and the underlying principles of AI risk auditing.

### QuLab Application Architecture Overview

The QuLab application follows a modular architecture using Streamlit for the user interface and a custom `source.py` module for core logic.

```mermaid
graph TD
    A[Raw Training Data (CSV)] --> B(Streamlit Application)
    B --> C{Session State Management};
    C --> D[Data Loading & Metadata Capture]
    D -- DatasetMetadata, DF, Schema --> E[Data Quality Checks]
    E -- DQ Findings --> F[Bias Assessment]
    F -- Bias Results --> G[Report Generation]
    G -- JSON Reports, Markdown Summary, Evidence Manifest --> H[Download Artifacts]
    H -- Auditable Output --> I(AI Governance Committee / Regulators)

    subgraph "Streamlit Core"
        B -- User Interactions --> B
        C -- State Preservation --> B
    end

    subgraph "source.py Module (Backend Logic)"
        D -- Pydantic Models --> D
        E -- Pandas, Numpy --> E
        F -- Fairness Calculations --> F
        G -- File I/O, Hashing --> G
    end
```
**Explanation:**
1.  **Raw Training Data**: The starting point is typically a CSV file containing the AI model's training data.
2.  **Streamlit Application**: The interactive web interface built with Streamlit.
3.  **Session State Management**: Streamlit's `st.session_state` is crucial for maintaining data and findings across user interactions and page navigations without re-running computations.
4.  **Data Loading & Metadata Capture**: Handles data ingestion, infers schema, and creates a `DatasetMetadata` object.
5.  **Data Quality Checks**: Performs various checks (missing values, duplicates, outliers, schema consistency) and generates `DataQualityFinding` objects.
6.  **Bias Assessment**: Calculates fairness metrics (DPD, DIR) across sensitive attributes and generates `BiasMetricResult` objects.
7.  **Report Generation**: Consolidates findings into JSON reports, a Markdown executive summary, and a cryptographic `EvidenceManifest`.
8.  **Download Artifacts**: Allows users to download all generated audit artifacts.
9.  **AI Governance Committee / Regulators**: The ultimate consumers of the auditable output for decision-making and compliance.

## Step 1: Setting up the Environment and Loading Data
Duration: 0:10

Sarah's first step is to prepare her analytical environment and load the dataset earmarked for audit. This ensures she has the necessary tools and the correct data to begin her investigation. For auditability and reproducibility, the dataset's metadata, including its source and ingestion date, must also be captured.

### Story + Context + Real-World Relevance

Imagine Sarah receiving a dataset for a new fraud detection model. Before any analysis begins, she needs to ensure the data is loaded correctly and that its origin is documented. If a critical issue is found later, knowing the `source_system` and `ingestion_date` allows her to trace back to the exact data pipeline or system that supplied the problematic data. This initial capture of metadata is fundamental for establishing **data provenance**, a cornerstone of explainable and auditable AI systems.

### Application Workflow: Data Loading

The Streamlit application provides two ways to load data:
1.  **Upload CSV**: Users can upload their own CSV file.
2.  **Load Synthetic Demo Data**: A button generates a synthetic dataset for demonstration purposes, mimicking fraud detection data.

After loading data, the user configures key parameters like the `label_column` and `sensitive_attributes`. These selections are critical for subsequent data quality and bias assessments.

**Streamlit UI Snippet:**
```python
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
```

Once the data is loaded, users are prompted to select the `label_column` (e.g., `is_fraud`) and `sensitive_attributes` (e.g., `customer_gender`, `customer_age_group`). These selections are stored in Streamlit's `st.session_state` to persist across pages.

```python
# Select label column
st.session_state.label_column = st.selectbox(
    "Select the Label Column (e.g., 'is_fraud')",
    options=all_columns,
    index=all_columns.index(default_label) if default_label in all_columns else 0,
    key="label_col_selector"
)

# Select sensitive attributes
st.session_state.sensitive_attributes = st.multiselect(
    "Select Sensitive Attributes (e.g., 'customer_gender', 'customer_age_group')",
    options=[col for col in all_columns if col != st.session_state.label_column],
    default=pre_selected_sensitive,
    key="sensitive_attr_selector"
)
```

Clicking **"Process Data & Capture Metadata"** triggers the inference of the dataset's schema and the creation of a `DatasetMetadata` object. This object, likely a Pydantic model from `source.py`, encapsulates crucial provenance information.

**Code Snippet: Metadata Capture**
```python
if st.button("Process Data & Capture Metadata"):
    with st.spinner("Capturing metadata and inferring schema..."):
        inferred_schema = {col: str(dtype) for col, dtype in st.session_state.df.dtypes.items()}
        
        # Example: Extracting source_system and ingestion_date from data if available
        source_system = st.session_state.df['source_system'].iloc[0] if 'source_system' in st.session_state.df.columns else "Unknown"
        ingestion_date_str = str(st.session_state.df['ingestion_date'].iloc[0]) if 'ingestion_date' in st.session_state.df.columns else datetime.now(timezone.utc).isoformat()
        
        try:
            ingestion_date = pd.to_datetime(ingestion_date_str)
        except Exception:
            st.warning(f"Could not parse 'ingestion_date' '{ingestion_date_str}'. Using current UTC datetime.")
            ingestion_date = datetime.now(timezone.utc)
        
        st.session_state.dataset_metadata = DatasetMetadata(
            source_system=source_system,
            ingestion_date=ingestion_date,
            schema=inferred_schema,
            label_column=st.session_state.label_column,
            sensitive_attributes=st.session_state.sensitive_attributes,
            record_count=len(st.session_state.df)
        )
        st.session_state.expected_schema = inferred_schema # Store for schema check
    st.success("Dataset metadata captured and schema inferred!")
    st.json(st.session_state.dataset_metadata.model_dump_json(indent=2))
```

<aside class="positive">
The `DatasetMetadata` object is crucial. It creates an immutable record of the data's state and origin at the time of the audit, ensuring **auditability** and **reproducibility**. The `model_dump_json()` method from Pydantic allows for easy serialization of this structured metadata.
</aside>

## Step 2: Conducting Data Quality Audit - Missingness and Duplicates
Duration: 0:15

Sarah begins her technical audit by checking for fundamental data quality issues. Missing values can lead to incomplete records, skewed statistics, or model errors if not handled properly. Duplicate records, if left unaddressed, can artificially inflate data volumes, bias model training, and lead to inaccurate performance metrics. Identifying these early ensures the model is trained on clean and reliable data.

### Story + Context + Real-World Relevance

Missing values in critical features can degrade model performance and lead to unfair decisions. For instance, if `customer_age` is frequently missing for certain demographic groups, the fraud detection model might perform poorly for those groups. Duplicate transactions could falsely represent activity, making certain fraud patterns appear more common than they are. Sarah must quantify these issues to understand their potential impact.

### Key Data Quality Metrics

1.  **Missing Value Rate**: The proportion of missing entries in a column.
    The **missing value rate** for a column $C$ in a dataset with $N$ records is given by:
    $$ \text{{Missing Rate}}(C) = \frac{{\text{{Number of Missing Values in }} C}}{{N}} $$
    where $C$ is a column and $N$ is the total number of records in the dataset.

2.  **Duplicate Row Count**: The number of rows that are exact copies of other rows.
    The **duplicate row count** for a dataset is simply the number of rows that are exact copies of other rows.

### Application Workflow: Missing Values and Duplicates

On the "Data Quality Audit" page, users can configure thresholds for missingness and duplicates. These thresholds define what is considered an acceptable level of data quality.

**Streamlit UI Snippet: Threshold Configuration**
```python
with st.expander("Missing Value Check Settings"):
    st.session_state.dq_threshold_missingness = st.number_input(
        "Missing Value Rate Threshold (e.g., 0.05 for 5%)",
        min_value=0.0, max_value=1.0, value=st.session_state.dq_threshold_missingness, step=0.01,
        format="%.3f", key="mv_thresh"
    )
with st.expander("Duplicate Check Settings"):
    st.session_state.dq_threshold_duplicates = st.number_input(
        "Duplicate Row Rate Threshold (e.001 for 0.1%)",
        min_value=0.0, max_value=1.0, value=st.session_state.dq_threshold_duplicates, step=0.001,
        format="%.4f", key="dup_thresh"
    )
```

When the user clicks **"Run All Data Quality Checks"**, the `perform_missing_value_check` and `perform_duplicate_check` functions (from `source.py`) are executed. These functions return a list of `DataQualityFinding` objects, each detailing a specific issue, its metric value, threshold, and status (PASS/FAIL).

**Code Snippet: Running Checks**
```python
# Inside "Run All Data Quality Checks" button logic
with st.spinner("Performing missing value checks..."):
    st.session_state.dq_findings.extend(
        perform_missing_value_check(st.session_state.df, threshold=st.session_state.dq_threshold_missingness)
    )
with st.spinner("Performing duplicate checks..."):
    st.session_state.dq_findings.append(
        perform_duplicate_check(st.session_state.df)
    )
# ... other checks will follow in next step
```

<aside class="positive">
Storing `dq_findings` in `st.session_state` ensures that the results are preserved and can be displayed, used for overall status calculation, and eventually included in the final reports. Each `DataQualityFinding` object provides a structured way to report issues consistently.
</aside>

## Step 3: Deepening Data Quality - Outliers and Schema Consistency
Duration: 0:15

Continuing her data quality audit, Sarah shifts her focus to identifying outliers and ensuring schema consistency. Outliers, extreme values in numerical features, can severely distort statistical analyses and model training, leading to inaccurate predictions or unstable models. Schema consistency ensures that the data types and structure align with expectations, preventing unexpected errors in downstream processing pipelines and model serving.

### Story + Context + Real-World Relevance

Outliers in transaction `amount` could represent legitimate high-value transactions, but they could also indicate data entry errors or even sophisticated fraud attempts. Incorrectly handling them can mislead the fraud model. Similarly, if a column expected to be numerical suddenly contains text (a schema drift), the model pipeline would break. Sarah needs to detect these issues proactively.

### Key Data Quality Metrics

1.  **Outlier Detection (IQR Method)**: Identifies values that fall significantly outside the interquartile range.
    For **outlier detection** using the Interquartile Range (IQR) method, a data point $x$ is considered an outlier if it falls outside the range:
    $$ [Q1 - k \times IQR, Q3 + k \times IQR] $$
    where $Q1$ is the first quartile, $Q3$ is the third quartile, $IQR = Q3 - Q1$ is the interquartile range, and $k$ is a multiplier (commonly 1.5).

2.  **Schema Consistency**: Verifies that column names and data types match an expected schema.
    **Schema consistency** is evaluated by comparing the inferred data types of each column in the loaded dataset against a predefined `expected_schema`.

### Application Workflow: Outliers and Schema Consistency

Users can configure the outlier rate threshold and the IQR multiplier ($k$) for outlier detection. The `expected_schema` is automatically inferred during the data loading step.

**Streamlit UI Snippet: Outlier Settings**
```python
with st.expander("Outlier Check Settings (IQR Method)"):
    st.session_state.dq_threshold_outliers = st.number_input(
        "Outlier Rate Threshold (e.g., 0.01 for 1%)",
        min_value=0.0, max_value=1.0, value=st.session_state.dq_threshold_outliers, step=0.001,
        format="%.3f", key="outlier_thresh"
    )
    iqr_multiplier = st.number_input(
        "IQR Multiplier (k in formula, e.g., 1.5)",
        min_value=1.0, max_value=3.0, value=1.5, step=0.1, key="iqr_multiplier"
    )
```

When **"Run All Data Quality Checks"** is clicked, the `perform_outlier_check_iqr` and `perform_schema_check` functions (from `source.py`) are executed.

**Code Snippet: Running Checks**
```python
# ... (after missing value and duplicate checks)
with st.spinner("Performing outlier checks..."):
    numerical_cols = st.session_state.df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if col not in ['transaction_id', st.session_state.label_column]:
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
    st.dataframe(dq_df[['issue_type', 'column', 'metric_value', 'threshold', 'status', 'description']])
    
    overall_dq_status = "PASS" if all(f.status == "PASS" for f in st.session_state.dq_findings if f.status != "N/A") else "FAIL"
    st.metric(label="Overall Data Quality Status", value=overall_dq_status)
```

The results are summarized in a DataFrame and an `Overall Data Quality Status` is displayed, indicating if any critical thresholds were breached.

## Step 4: Assessing Algorithmic Bias - Demographic Parity and Disparate Impact
Duration: 0:20

A core part of Sarah's role as a Data Risk Lead is to ensure that GlobalTrust Financial's AI systems do not perpetuate or amplify existing societal biases. Unfair outcomes can lead to regulatory scrutiny, legal challenges, and severe reputational damage. She must quantify potential biases in the *TransactionGuard AI* training data, particularly concerning sensitive attributes like `customer_gender`, `customer_age_group`, and `customer_region`.

### Story + Context + Real-World Relevance

If the fraud detection model disproportionately flags transactions from specific demographic groups as fraudulent (even when the underlying behavior isn't inherently riskier), it indicates bias in the training data. This could lead to legitimate customers from these groups facing unnecessary transaction declines or additional scrutiny, creating a discriminatory experience. Sarah uses formal fairness metrics to objectively measure these disparities.

### Key Fairness Metrics

1.  **Demographic Parity Difference (DPD)**: Measures the difference in the positive outcome rate between a protected group and a reference group. An ideal DPD is 0, indicating equal outcomes.
    $$ DPD = P(\text{{Positive Outcome}} | \text{{Protected Group}}) - P(\text{{Positive Outcome}} | \text{{Reference Group}}) $$
    where $P(\text{{Positive Outcome}} | \text{{Group}})$ is the probability of a positive outcome for a given group.
    A common threshold for DPD is that its absolute value should be less than 0.1, i.e., $ |DPD| \le 0.1 $.

2.  **Disparate Impact Ratio (DIR)**: Measures the ratio of the positive outcome rate of a protected group to that of a reference group. An ideal DIR is 1.0, indicating equal outcomes.
    $$ DIR = \frac{{P(\text{{Positive Outcome}} | \text{{Protected Group}})}}{{P(\text{{Positive Outcome}} | \text{{Reference Group}})}} $$
    where $P(\text{{Positive Outcome}} | \text{{Group}})$ is the probability of a positive outcome for a given group.
    A common threshold for DIR is the '80% rule,' where $ 0.8 \le DIR \le 1.25 $.

### Application Workflow: Bias Assessment

The "Bias Assessment" page requires the user to specify the `POSITIVE_LABEL_VALUE` for the target column (e.g., `1` for fraud). Crucially, for each selected sensitive attribute, a **reference group** must be chosen. This reference group serves as the baseline against which other "protected" groups are compared.

**Streamlit UI Snippet: Bias Configuration**
```python
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
```

Users also configure the acceptable threshold ranges for DPD and DIR. These thresholds determine whether a detected disparity is flagged as a PASS or FAIL.

**Streamlit UI Snippet: Bias Thresholds**
```python
with st.expander("Demographic Parity Difference (DPD) Thresholds"):
    st.session_state.bias_dpd_threshold_min = st.number_input(...)
    st.session_state.bias_dpd_threshold_max = st.number_input(...)

with st.expander("Disparate Impact Ratio (DIR) Thresholds"):
    st.session_state.bias_dir_threshold_min = st.number_input(...)
    st.session_state.bias_dir_threshold_max = st.number_input(...)
```

Clicking **"Run Bias Assessment"** triggers the calculation of DPD and DIR for each protected group relative to its chosen reference group within each sensitive attribute. The `calculate_demographic_parity_difference` and `calculate_disparate_impact_ratio` functions (from `source.py`) return a list of `BiasMetricResult` objects.

**Code Snippet: Running Bias Assessment**
```python
if st.button("Run Bias Assessment"):
    st.session_state.bias_results = [] # Reset results
    
    for attr in st.session_state.sensitive_attributes:
        if attr in st.session_state.reference_groups:
            ref_group = st.session_state.reference_groups[attr]
            
            # Calculate DPD
            dpd_results = calculate_demographic_parity_difference(
                st.session_state.df, attr, st.session_state.label_column,
                POSITIVE_LABEL_VALUE, ref_group,
                threshold_min=st.session_state.bias_dpd_threshold_min,
                threshold_max=st.session_state.bias_dpd_threshold_max
            )
            st.session_state.bias_results.extend(dpd_results)
            
            # Calculate DIR
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
```

<aside class="negative">
Bias in training data is a serious risk. If `BiasMetricResult` objects show "FAIL" statuses for DPD or DIR, it indicates that the training data itself exhibits problematic disparities, which the AI model will likely learn and amplify, leading to unfair outcomes.
</aside>

## Step 5: Generating Auditable Reports and Evidence Manifest
Duration: 0:10

Having performed the detailed quality and bias checks, Sarah's next crucial step is to consolidate all findings into structured, auditable reports. This is not just about summarizing data; it's about creating verifiable artifacts that can be presented to internal governance committees and external regulators.

### Story + Context + Real-World Relevance

For GlobalTrust Financial, every AI system deployment requires a clear audit trail. Sarah's reports serve as the official record of the pre-deployment audit. The `data_quality_report.json` and `bias_metrics.json` provide granular details, while the `evidence_manifest.json` acts as a cryptographic fingerprint of the entire audit, linking findings to the specific input data and tools used. This level of detail is paramount for demonstrating compliance and accountability to stakeholders and regulators.

### Core Audit Artifacts

1.  **Data Quality Report (`data_quality_report.json`)**: A JSON file containing all `DataQualityFinding` objects, providing a detailed breakdown of data quality issues.
2.  **Bias Metrics Report (`bias_metrics.json`)**: A JSON file containing all `BiasMetricResult` objects, detailing fairness metric values, thresholds, and statuses for each sensitive group.
3.  **Executive Summary (`case_executive_summary.md`)**: A human-readable Markdown report summarizing key findings and recommendations (covered in Step 6).
4.  **Evidence Manifest (`evidence_manifest.json`)**: A cryptographic manifest that ties all audit artifacts together, ensuring provenance and integrity.

The `inputs_hash` is calculated by hashing the entire raw input dataset. The `outputs_hash` is a dictionary of SHA256 hashes for each generated report file.
$$ \text{{SHA256 Hash}} = \text{{hashlib.sha256}}(\text{{data.encode()}}).\text{{hexdigest}}() $$
where `data` is the string representation of the content to be hashed.

### Application Workflow: Report Generation

The "Reports & Recommendations" page facilitates the generation of all these artifacts. When the "Generate All Reports and Evidence Manifest" button is clicked, the application calls various functions from `source.py` to create the JSON and Markdown files.

**Code Snippet: Report Generation**
```python
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
        # Initial manifest creation (hash the raw input data and initial reports)
        st.session_state.evidence_manifest_hash = generate_evidence_manifest_json(
            df=st.session_state.df,
            dq_findings_hash=st.session_state.dq_report_hash,
            bias_results_hash=st.session_state.bias_report_hash,
            dq_report_filename=DATA_QUALITY_REPORT_FILENAME,
            bias_report_filename=BIAS_METRICS_REPORT_FILENAME,
            output_path=os.path.join(st.session_state.output_dir, EVIDENCE_MANIFEST_FILENAME)
        )
        
        # Manually update manifest with executive summary, then re-hash manifest itself
        # This reflects that the summary is also an auditable artifact
        evidence_manifest_path = os.path.join(st.session_state.output_dir, EVIDENCE_MANIFEST_FILENAME)
        with open(evidence_manifest_path, 'r') as f:
            manifest_data = json.load(f)
        
        if EXECUTIVE_SUMMARY_FILENAME not in manifest_data['outputs_hash']:
            manifest_data['outputs_hash'][EXECUTIVE_SUMMARY_FILENAME] = st.session_state.executive_summary_hash
            # Add artifact details (assuming Artifact is a Pydantic model from source.py)
            manifest_data['artifacts'].append(
                Artifact(file_name=EXECUTIVE_SUMMARY_FILENAME, 
                         file_hash=st.session_state.executive_summary_hash, 
                         description="Consolidated executive summary of audit findings and recommendations.").model_dump()
            )
        
        manifest_content_updated = json.dumps(manifest_data, indent=4)
        with open(evidence_manifest_path, 'w') as f:
            f.write(manifest_content_updated)
        st.session_state.evidence_manifest_hash = calculate_sha256_hash(manifest_content_updated)

    st.success("All reports and evidence manifest generated!")
```

After generation, the SHA256 hashes of each report and the overall manifest are displayed. These hashes serve as cryptographic fingerprints, ensuring the integrity of the audit process.

<aside class="positive">
The `evidence_manifest.json` is the backbone of the audit trail. By storing hashes of both the input data and all generated reports, it provides an unalterable record of the audit, crucial for **compliance** and demonstrating **accountability**.
</aside>

## Step 6: Reviewing the Executive Summary and Mitigation Recommendations
Duration: 0:05

The final and most critical step for Sarah is to synthesize all audit findings into a concise executive summary for the AI Governance Committee. This document must clearly articulate the key risks identified, provide a summary PASS/FAIL status, and propose actionable, testable mitigation recommendations. This is where technical findings are translated into strategic insights for leadership.

### Story + Context + Real-World Relevance

Sarah understands that the AI Governance Committee needs a high-level overview, not raw data. Her executive summary must highlight the most significant data quality and bias risks, explaining their potential impact on GlobalTrust and its customers. More importantly, she must provide concrete recommendations that her team or data engineering can implement to resolve these issues before the *TransactionGuard AI* model goes live. This direct connection from problem to solution demonstrates proactive risk management.

### The Executive Summary

The `case_executive_summary.md` file (generated in the previous step) is now presented in the Streamlit application. This Markdown file is crafted to be easily readable and understandable by non-technical stakeholders. It aggregates the overall data quality status, critical bias findings, and offers concrete steps for remediation.

**Streamlit UI Snippet: Executive Summary Display and Download**
```python
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
```

Finally, to streamline the process of sharing and archiving audit results, the application provides a button to download all generated audit artifacts (JSON reports, Executive Summary, Evidence Manifest) as a single ZIP file.

**Streamlit UI Snippet: Download All Artifacts**
```python
st.subheader("Download All Audit Artifacts")
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(st.session_state.output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            zf.write(file_path, arcname=os.path.relpath(file_path, st.session_state.output_dir))

zip_buffer.seek(0)

st.download_button(
    label="Download All Audit Artifacts (ZIP)",
    data=zip_buffer.getvalue(),
    file_name="audit_artifacts.zip",
    mime="application/zip",
    help="Includes data_quality_report.json, bias_metrics.json, case_executive_summary.md, and evidence_manifest.json"
)
```

<aside class="positive">
The Executive Summary translates complex technical findings into actionable business insights, making it a powerful tool for strategic decision-making. Providing a single ZIP download for all artifacts enhances convenience and ensures that a complete, auditable package is available for stakeholders.
</aside>

This concludes the QuLab codelab. You've now explored the essential functionalities of the application for conducting comprehensive AI risk audits, from data ingestion and quality checks to bias assessment and auditable report generation.
