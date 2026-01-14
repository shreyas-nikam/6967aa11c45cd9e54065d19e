
# Pre-Deployment AI System Risk Audit: Fraud Detection Model Training Data

## Case Study Introduction

As **Sarah Chen, the Data Risk Lead** at **GlobalTrust Financial**, your primary responsibility is to safeguard the integrity and fairness of the institution's AI systems. GlobalTrust is on the cusp of deploying a new AI-powered fraud detection model, *TransactionGuard AI*, a high-stakes system designed to protect customers and the institution from financial crime. However, before it can go live, it must undergo a rigorous pre-deployment audit.

Your task is to conduct a comprehensive assessment of the model's training data. This audit is crucial for identifying potential data quality issues, uncovering problematic biases against sensitive customer groups, and verifying data provenance. Failing to identify these risks upfront could lead to severe consequences: unfair outcomes for customers, substantial regulatory fines, reputational damage, and erosion of trust.

This notebook simulates your workflow as you systematically analyze the *TransactionGuard AI* training dataset, generating the necessary evidence and reports for the AI Governance Committee. Your findings will directly inform the decision to deploy the model or necessitate further remediation.

---

## 1. Environment Setup and Data Loading

Sarah's first step is always to prepare her analytical environment and load the dataset earmarked for audit. This ensures she has the necessary tools and the correct data to begin her investigation. She needs to ensure her environment has all the required libraries and then load the specific training data for the *TransactionGuard AI* model. For auditability and reproducibility, the dataset's metadata, including its source and ingestion date, must also be captured.

### 1.1 Install Required Libraries

```python
!pip install pandas numpy scikit-learn pydantic hashlib uuid datetime
```

### 1.2 Import Dependencies and Define Data Models

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum
import uuid
from datetime import datetime, timezone
import hashlib
import json
import os

# --- Pydantic Data Models (as per specification) ---

class DataQualityIssue(str, Enum):
    MISSINGNESS = "MISSINGNESS"
    OUTLIERS = "OUTLIERS"
    DUPLICATES = "DUPLICATES"
    SCHEMA_DRIFT = "SCHEMA_DRIFT"

class BiasMetric(str, Enum):
    DEMOGRAPHIC_PARITY = "DEMOGRAPHIC_PARITY"
    DISPARATE_IMPACT = "DISPARATE_IMPACT"

class DatasetMetadata(BaseModel):
    dataset_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_system: str
    ingestion_date: datetime
    schema: Dict[str, str] # column: type_as_string
    label_column: str
    sensitive_attributes: List[str]
    record_count: int

class DataQualityFinding(BaseModel):
    issue_type: DataQualityIssue
    column: Optional[str] = None
    metric_value: float
    threshold: float
    status: str # "PASS" or "FAIL"
    description: str

class BiasMetricResult(BaseModel):
    metric_name: BiasMetric
    sensitive_attribute: str
    group: str
    reference_group: str
    value: float
    threshold_min: float
    threshold_max: float
    status: str # "PASS" or "FAIL"
    interpretation: str

class Artifact(BaseModel):
    file_name: str
    file_hash: str # SHA256 hash
    description: str

class EvidenceManifest(BaseModel):
    run_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    team_or_user: str = "Sarah Chen (Data Risk Lead)"
    app_version: str = "1.0.0"
    inputs_hash: str # Hash of the raw input data
    outputs_hash: Dict[str, str] # Hashes of generated artifacts (filename: hash)
    artifacts: List[Artifact]

```

### 1.3 Generate Synthetic Training Data and Extract Metadata

Sarah doesn't rely on external file uploads for the audit process in a notebook. Instead, for explicit reproducibility, she generates a synthetic dataset that mimics the characteristics of fraud detection training data, including known issues she expects to find. This generated dataset will serve as the *TransactionGuard AI* model's training data.

```python
def generate_synthetic_fraud_data(num_records=10000):
    np.random.seed(42)
    
    data = {
        'transaction_id': [str(uuid.uuid4()) for _ in range(num_records)],
        'amount': np.random.normal(loc=100, scale=50, size=num_records),
        'customer_age': np.random.randint(18, 90, size=num_records),
        'customer_gender': np.random.choice(['Male', 'Female', 'Non-Binary'], size=num_records, p=[0.48, 0.48, 0.04]),
        'customer_region': np.random.choice(['North', 'South', 'East', 'West'], size=num_records, p=[0.25, 0.25, 0.25, 0.25]),
        'merchant_category': np.random.choice(['Retail', 'Online', 'Services', 'Travel'], size=num_records),
        'transaction_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, size=num_records), unit='D'),
        'source_system': ['CoreBanking_ETL'] * num_records,
        'ingestion_date': [pd.to_datetime('2023-01-15')] * num_records,
        'is_fraud': np.random.choice([0, 1], size=num_records, p=[0.95, 0.05]) # 5% fraud rate
    }

    df = pd.DataFrame(data)

    # Introduce specific issues for audit demonstration
    # 1. Missing values in customer_age
    missing_indices = np.random.choice(df.index, size=int(0.06 * num_records), replace=False) # 6% missing
    df.loc[missing_indices, 'customer_age'] = np.nan

    # 2. Duplicate rows
    duplicate_indices = np.random.choice(df.index, size=int(0.01 * num_records), replace=False) # 1% duplicates
    df = pd.concat([df, df.loc[duplicate_indices]]).reset_index(drop=True)

    # 3. Outliers in amount
    outlier_indices = np.random.choice(df.index, size=int(0.005 * num_records), replace=False) # 0.5% outliers
    df.loc[outlier_indices, 'amount'] = np.random.uniform(low=1000, high=5000, size=len(outlier_indices))

    # 4. Introduce bias: Make fraud slightly more likely for 'Male' and 'Age Group: 18-30'
    # Create age groups first
    bins = [0, 30, 50, 70, 100]
    labels = ['18-30', '31-50', '51-70', '71+']
    df['customer_age_group'] = pd.cut(df['customer_age'], bins=bins, labels=labels, right=True, include_lowest=True)
    df['customer_age_group'] = df['customer_age_group'].astype(str) # Convert to string for consistent handling with sensitive attributes

    male_18_30_indices = df[(df['customer_gender'] == 'Male') & (df['customer_age_group'] == '18-30') & (df['is_fraud'] == 0)].index
    fraud_bias_indices = np.random.choice(male_18_30_indices, size=int(0.02 * len(male_18_30_indices)), replace=False)
    df.loc[fraud_bias_indices, 'is_fraud'] = 1 # Increase fraud rate for this group

    # Finalize dtypes and ensure consistent representation
    df['is_fraud'] = df['is_fraud'].astype(int)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['ingestion_date'] = pd.to_datetime(df['ingestion_date'])

    return df

# Generate the dataset
training_data = generate_synthetic_fraud_data(num_records=10000)

# Define audit parameters
LABEL_COLUMN = 'is_fraud'
SENSITIVE_ATTRIBUTES = ['customer_gender', 'customer_age_group', 'customer_region']
EXPECTED_SCHEMA = {
    'transaction_id': 'object',
    'amount': 'float64',
    'customer_age': 'float64',
    'customer_gender': 'object',
    'customer_region': 'object',
    'merchant_category': 'object',
    'transaction_date': 'datetime64[ns]',
    'source_system': 'object',
    'ingestion_date': 'datetime64[ns]',
    'is_fraud': 'int64',
    'customer_age_group': 'object'
}

# Capture dataset metadata
inferred_schema = {col: str(dtype) for col, dtype in training_data.dtypes.items()}
dataset_metadata = DatasetMetadata(
    source_system=training_data['source_system'].iloc[0],
    ingestion_date=training_data['ingestion_date'].iloc[0],
    schema=inferred_schema,
    label_column=LABEL_COLUMN,
    sensitive_attributes=SENSITIVE_ATTRIBUTES,
    record_count=len(training_data)
)

print(f"Dataset loaded with {len(training_data)} records.")
print("\nFirst 5 rows of the dataset:")
print(training_data.head())
print("\nDataset Metadata:")
print(dataset_metadata.model_dump_json(indent=2))

```

#### Explanation of Execution: Data Loading

Sarah's first review confirms that the synthetic data has been generated and loaded correctly, reflecting typical financial transaction data. The captured `DatasetMetadata` provides a vital snapshot of the data's structure and origin (`source_system`, `ingestion_date`), which is essential for auditability and understanding lineage risks. For instance, knowing the data comes from 'CoreBanking_ETL' and was ingested on '2023-01-15' allows her to trace back any fundamental issues to a specific batch or system. The auto-inferred schema is crucial for the upcoming schema consistency checks.

---

## 2. Data Quality Audit: Missing Values and Duplicates

Sarah begins her technical audit by checking for fundamental data quality issues. Missing values can lead to incomplete records, skewed statistics, or model errors if not handled properly. Duplicate records, if left unaddressed, can artificially inflate data volumes, bias model training, and lead to inaccurate performance metrics. Identifying these early ensures the model is trained on clean and reliable data.

### 2.1 Story + Context + Real-World Relevance

Missing values in critical features can degrade model performance and lead to unfair decisions. For instance, if `customer_age` is frequently missing for certain demographic groups, the fraud detection model might perform poorly for those groups. Duplicate transactions could falsely represent activity, making certain fraud patterns appear more common than they are. Sarah must quantify these issues to understand their potential impact.

The **missing value rate** for a column $C$ in a dataset with $N$ records is given by:
$$ \text{Missing Rate}(C) = \frac{\text{Number of Missing Values in } C}{N} $$

The **duplicate row count** for a dataset is simply the number of rows that are exact copies of other rows.

### 2.2 Code Cell (Function Definition + Function Execution)

```python
def perform_missing_value_check(df: pd.DataFrame, threshold: float = 0.05) -> List[DataQualityFinding]:
    """
    Performs a missing value check on the DataFrame and returns DataQualityFinding objects.
    Threshold is the maximum acceptable missing percentage.
    """
    findings = []
    for column in df.columns:
        missing_rate = df[column].isnull().sum() / len(df)
        status = "PASS" if missing_rate <= threshold else "FAIL"
        description = f"Column '{column}' has {missing_rate:.2%} missing values."
        if status == "FAIL":
            description += f" This exceeds the threshold of {threshold:.2%}. High risk of data incompleteness."
        
        findings.append(DataQualityFinding(
            issue_type=DataQualityIssue.MISSINGNESS,
            column=column,
            metric_value=missing_rate,
            threshold=threshold,
            status=status,
            description=description
        ))
    return findings

def perform_duplicate_check(df: pd.DataFrame) -> DataQualityFinding:
    """
    Performs a duplicate row check on the DataFrame and returns a DataQualityFinding object.
    """
    num_duplicates = df.duplicated().sum()
    total_records = len(df)
    duplicate_rate = num_duplicates / total_records
    # Define a low threshold for duplicates, e.g., 0.1% or 0.001
    threshold = 0.001 
    status = "PASS" if duplicate_rate <= threshold else "FAIL"
    description = f"Dataset contains {num_duplicates} duplicate rows, which is {duplicate_rate:.2%} of total records."
    if status == "FAIL":
        description += f" This exceeds the threshold of {threshold:.2%}. Potential for biased model training."

    return DataQualityFinding(
        issue_type=DataQualityIssue.DUPLICATES,
        column=None, # Duplicates apply to the whole dataset, not a single column
        metric_value=duplicate_rate,
        threshold=threshold,
        status=status,
        description=description
    )

# Execute the data quality checks
dq_findings = []
dq_findings.extend(perform_missing_value_check(training_data, threshold=0.05)) # 5% missingness threshold
dq_findings.append(perform_duplicate_check(training_data))

print("--- Missing Value Check Results ---")
for finding in [f for f in dq_findings if f.issue_type == DataQualityIssue.MISSINGNESS]:
    print(f"[{finding.status}] {finding.column}: {finding.description}")

print("\n--- Duplicate Check Results ---")
for finding in [f for f in dq_findings if f.issue_type == DataQualityIssue.DUPLICATES]:
    print(f"[{finding.status}] {finding.description}")

```

### 2.3 Explanation of Execution: Missing Values and Duplicates

The audit reveals that the `customer_age` column has a significant missing value rate (likely 6% as designed), exceeding GlobalTrust's 5% threshold, resulting in a **FAIL** status. Sarah knows this means imputations or advanced missing value handling strategies are required before training, as simply dropping these rows could remove valuable data or introduce further bias.

Furthermore, the check identified duplicate rows (likely 1% as designed), also exceeding the low duplicate threshold. This finding is a **FAIL** and signals that the dataset needs de-duplication. Ignoring duplicates would lead to the model over-learning from redundant data points, potentially skewing feature importance and performance metrics. These initial findings are critical for Sarah to flag for the data engineering team.

---

## 3. Data Quality Audit: Outliers and Schema Consistency

Continuing her data quality audit, Sarah shifts her focus to identifying outliers and ensuring schema consistency. Outliers, extreme values in numerical features, can severely distort statistical analyses and model training, leading to inaccurate predictions or unstable models. Schema consistency ensures that the data types and structure align with expectations, preventing unexpected errors in downstream processing pipelines and model serving.

### 3.1 Story + Context + Real-World Relevance

Outliers in transaction `amount` could represent legitimate high-value transactions, but they could also indicate data entry errors or even sophisticated fraud attempts. Incorrectly handling them can mislead the fraud model. Similarly, if a column expected to be numerical suddenly contains text (a schema drift), the model pipeline would break. Sarah needs to detect these issues proactively.

For **outlier detection** using the Interquartile Range (IQR) method, a data point $x$ is considered an outlier if it falls outside the range:
$$ [Q1 - k \times IQR, Q3 + k \times IQR] $$
where $Q1$ is the first quartile, $Q3$ is the third quartile, $IQR = Q3 - Q1$ is the interquartile range, and $k$ is a multiplier (commonly 1.5).

**Schema consistency** is evaluated by comparing the inferred data types of each column in the loaded dataset against a predefined `expected_schema`.

### 3.2 Code Cell (Function Definition + Function Execution)

```python
def perform_outlier_check_iqr(df: pd.DataFrame, column: str, threshold_iqr_multiplier: float = 1.5) -> DataQualityFinding:
    """
    Performs outlier detection using the IQR method for a given numerical column.
    Returns a DataQualityFinding object.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return DataQualityFinding(
            issue_type=DataQualityIssue.OUTLIERS,
            column=column,
            metric_value=np.nan,
            threshold=threshold_iqr_multiplier,
            status="N/A",
            description=f"Outlier check skipped for non-numeric column '{column}'."
        )

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold_iqr_multiplier * IQR
    upper_bound = Q3 + threshold_iqr_multiplier * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    num_outliers = len(outliers)
    outlier_rate = num_outliers / len(df)
    
    # Define a low threshold for outlier rate, e.g., 1% or 0.01
    outlier_threshold = 0.01 
    status = "PASS" if outlier_rate <= outlier_threshold else "FAIL"
    description = (
        f"Column '{column}' has {num_outliers} outliers ({outlier_rate:.2%}) based on IQR multiplier {threshold_iqr_multiplier}. "
        f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]."
    )
    if status == "FAIL":
        description += f" This exceeds the threshold of {outlier_threshold:.2%}. Potential for distorted model training."

    return DataQualityFinding(
        issue_type=DataQualityIssue.OUTLIERS,
        column=column,
        metric_value=outlier_rate,
        threshold=outlier_threshold,
        status=status,
        description=description
    )

def perform_schema_check(df: pd.DataFrame, expected_schema: Dict[str, str]) -> List[DataQualityFinding]:
    """
    Compares the DataFrame's inferred schema against an expected schema.
    Returns DataQualityFinding objects for any mismatches.
    """
    findings = []
    inferred_schema = {col: str(df[col].dtype) for col in df.columns}

    # Check for missing columns in dataframe compared to expected
    for col in expected_schema:
        if col not in inferred_schema:
            findings.append(DataQualityFinding(
                issue_type=DataQualityIssue.SCHEMA_DRIFT,
                column=col,
                metric_value=1.0, # Represents a complete mismatch (missing column)
                threshold=0.0, # Any missing column is a fail
                status="FAIL",
                description=f"Column '{col}' expected in schema but not found in dataset."
            ))
            
    # Check for type mismatches or extra columns
    for col, inferred_type in inferred_schema.items():
        if col not in expected_schema:
            findings.append(DataQualityFinding(
                issue_type=DataQualityIssue.SCHEMA_DRIFT,
                column=col,
                metric_value=1.0, # Represents a complete mismatch (extra column)
                threshold=0.0,
                status="FAIL",
                description=f"Column '{col}' found in dataset but not in expected schema (inferred type: {inferred_type})."
            ))
        elif inferred_type != expected_schema[col]:
            findings.append(DataQualityFinding(
                issue_type=DataQualityIssue.SCHEMA_DRIFT,
                column=col,
                metric_value=0.5, # Represents a type mismatch
                threshold=0.0, # Any type mismatch is a fail
                status="FAIL",
                description=f"Schema mismatch for column '{col}': Expected '{expected_schema[col]}', found '{inferred_type}'."
            ))
        else:
            findings.append(DataQualityFinding(
                issue_type=DataQualityIssue.SCHEMA_DRIFT,
                column=col,
                metric_value=0.0,
                threshold=0.0,
                status="PASS",
                description=f"Schema for column '{col}' matches expected type '{expected_schema[col]}'."
            ))
    return findings


# Execute the data quality checks
dq_findings.append(perform_outlier_check_iqr(training_data, 'amount', threshold_iqr_multiplier=1.5))
dq_findings.extend(perform_schema_check(training_data, EXPECTED_SCHEMA))

print("--- Outlier Check Results ---")
for finding in [f for f in dq_findings if f.issue_type == DataQualityIssue.OUTLIERS]:
    print(f"[{finding.status}] {finding.column}: {finding.description}")

print("\n--- Schema Consistency Check Results ---")
for finding in [f for f in dq_findings if f.issue_type == DataQualityIssue.SCHEMA_DRIFT]:
    print(f"[{finding.status}] {finding.column if finding.column else 'Dataset'}: {finding.description}")
```

### 3.3 Explanation of Execution: Outliers and Schema Consistency

The outlier check for the `amount` column likely reports a **FAIL** status (as designed). Sarah notes that these outliers, representing unusually high transaction values, could either be legitimate but rare events, or indicators of sophisticated fraudulent activity or data errors. This requires further investigation by her team. Depending on the nature of these outliers, strategies likeWinsorization or robust scaling might be necessary before model training to prevent the model from being overly influenced by these extreme values.

The schema consistency check should report mostly **PASS** statuses for the columns if `EXPECTED_SCHEMA` matches the generated data types. If there were any mismatches, say `amount` was inferred as `object` instead of `float64`, it would flag a **FAIL**. Sarah understands that schema drift is a critical issue that can silently break production models, making this check vital for preventing deployment failures. The results confirm that the data types are as expected, reassuring her that the data will fit the model's input requirements.

---

## 4. Bias Assessment: Demographic Parity and Disparate Impact

A core part of Sarah's role as a Data Risk Lead is to ensure that GlobalTrust Financial's AI systems do not perpetuate or amplify existing societal biases. Unfair outcomes can lead to regulatory scrutiny, legal challenges, and severe reputational damage. She must quantify potential biases in the *TransactionGuard AI* training data, particularly concerning sensitive attributes like `customer_gender`, `customer_age_group`, and `customer_region`.

### 4.1 Story + Context + Real-World Relevance

If the fraud detection model disproportionately flags transactions from specific demographic groups as fraudulent (even when the underlying behavior isn't inherently riskier), it indicates bias in the training data. This could lead to legitimate customers from these groups facing unnecessary transaction declines or additional scrutiny, creating a discriminatory experience. Sarah uses formal fairness metrics to objectively measure these disparities.

**Demographic Parity Difference (DPD)** measures the difference in the positive outcome rate (e.g., fraud detected) between a protected group and a reference group. An ideal DPD is 0, indicating equal outcomes.
$$ DPD = P(\text{Positive Outcome} | \text{Protected Group}) - P(\text{Positive Outcome} | \text{Reference Group}) $$
A common threshold for DPD is that its absolute value should be less than 0.1, i.e., $ |DPD| \le 0.1 $.

**Disparate Impact Ratio (DIR)** measures the ratio of the positive outcome rate of a protected group to that of a reference group. An ideal DIR is 1.0, indicating equal outcomes.
$$ DIR = \frac{P(\text{Positive Outcome} | \text{Protected Group})}{P(\text{Positive Outcome} | \text{Reference Group})} $$
A common threshold for DIR is the "80% rule," where $ 0.8 \le DIR \le 1.25 $.

### 4.2 Code Cell (Function Definition + Function Execution)

```python
def calculate_demographic_parity_difference(
    df: pd.DataFrame, 
    sensitive_attr: str, 
    label_col: str, 
    positive_label: int, 
    reference_group: str,
    threshold_min: float = -0.1, 
    threshold_max: float = 0.1
) -> List[BiasMetricResult]:
    """
    Calculates Demographic Parity Difference for each group within a sensitive attribute.
    Returns BiasMetricResult objects.
    """
    results = []
    
    # Ensure sensitive_attr is treated as categorical
    df[sensitive_attr] = df[sensitive_attr].astype(str)

    # Calculate overall positive outcome rate for the reference group
    ref_group_df = df[df[sensitive_attr] == reference_group]
    if len(ref_group_df) == 0:
        return [BiasMetricResult(
            metric_name=BiasMetric.DEMOGRAPHIC_PARITY,
            sensitive_attribute=sensitive_attr,
            group=reference_group,
            reference_group=reference_group,
            value=np.nan,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            status="ERROR",
            interpretation=f"Reference group '{reference_group}' not found in sensitive attribute '{sensitive_attr}'."
        )]
    
    ref_positive_rate = (ref_group_df[label_col] == positive_label).mean()

    for group in df[sensitive_attr].unique():
        if group == reference_group:
            # Reference group's own DPD is 0 by definition
            interpretation = f"Reference group for {sensitive_attr}. DPD is 0 by definition."
            status = "PASS" # Always passes for reference
            results.append(BiasMetricResult(
                metric_name=BiasMetric.DEMOGRAPHIC_PARITY,
                sensitive_attribute=sensitive_attr,
                group=str(group),
                reference_group=reference_group,
                value=0.0,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                status=status,
                interpretation=interpretation
            ))
            continue

        group_df = df[df[sensitive_attr] == group]
        if len(group_df) == 0:
            results.append(BiasMetricResult(
                metric_name=BiasMetric.DEMOGRAPHIC_PARITY,
                sensitive_attribute=sensitive_attr,
                group=str(group),
                reference_group=reference_group,
                value=np.nan,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                status="ERROR",
                interpretation=f"Group '{group}' has no records in sensitive attribute '{sensitive_attr}'."
            ))
            continue
            
        group_positive_rate = (group_df[label_col] == positive_label).mean()
        dpd = group_positive_rate - ref_positive_rate
        
        status = "PASS" if threshold_min <= dpd <= threshold_max else "FAIL"
        interpretation = (
            f"The positive outcome rate for '{group}' is {group_positive_rate:.2%} compared to {ref_positive_rate:.2%} for '{reference_group}'. "
            f"DPD of {dpd:.3f} indicates {'no significant disparity' if status == 'PASS' else 'a significant disparity'} in positive outcomes."
        )
        
        results.append(BiasMetricResult(
            metric_name=BiasMetric.DEMOGRAPHIC_PARITY,
            sensitive_attribute=sensitive_attr,
            group=str(group),
            reference_group=reference_group,
            value=dpd,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            status=status,
            interpretation=interpretation
        ))
    return results

def calculate_disparate_impact_ratio(
    df: pd.DataFrame, 
    sensitive_attr: str, 
    label_col: str, 
    positive_label: int, 
    reference_group: str,
    threshold_min: float = 0.8, 
    threshold_max: float = 1.25
) -> List[BiasMetricResult]:
    """
    Calculates Disparate Impact Ratio for each group within a sensitive attribute.
    Returns BiasMetricResult objects.
    """
    results = []

    # Ensure sensitive_attr is treated as categorical
    df[sensitive_attr] = df[sensitive_attr].astype(str)

    # Calculate overall positive outcome rate for the reference group
    ref_group_df = df[df[sensitive_attr] == reference_group]
    if len(ref_group_df) == 0:
        return [BiasMetricResult(
            metric_name=BiasMetric.DISPARATE_IMPACT,
            sensitive_attribute=sensitive_attr,
            group=reference_group,
            reference_group=reference_group,
            value=np.nan,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            status="ERROR",
            interpretation=f"Reference group '{reference_group}' not found in sensitive attribute '{sensitive_attr}'."
        )]
    
    ref_positive_rate = (ref_group_df[label_col] == positive_label).mean()

    for group in df[sensitive_attr].unique():
        if group == reference_group:
            # Reference group's own DIR is 1 by definition
            interpretation = f"Reference group for {sensitive_attr}. DIR is 1 by definition."
            status = "PASS" # Always passes for reference
            results.append(BiasMetricResult(
                metric_name=BiasMetric.DISPARATE_IMPACT,
                sensitive_attribute=sensitive_attr,
                group=str(group),
                reference_group=reference_group,
                value=1.0,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                status=status,
                interpretation=interpretation
            ))
            continue

        group_df = df[df[sensitive_attr] == group]
        if len(group_df) == 0:
            results.append(BiasMetricResult(
                metric_name=BiasMetric.DISPARATE_IMPACT,
                sensitive_attribute=sensitive_attr,
                group=str(group),
                reference_group=reference_group,
                value=np.nan,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                status="ERROR",
                interpretation=f"Group '{group}' has no records in sensitive attribute '{sensitive_attr}'."
            ))
            continue

        group_positive_rate = (group_df[label_col] == positive_label).mean()
        
        # Handle division by zero if reference group has 0 positive rate
        if ref_positive_rate == 0:
            dir_value = np.inf if group_positive_rate > 0 else 1.0
        else:
            dir_value = group_positive_rate / ref_positive_rate
        
        status = "PASS" if threshold_min <= dir_value <= threshold_max else "FAIL"
        interpretation = (
            f"The positive outcome rate for '{group}' is {group_positive_rate:.2%} compared to {ref_positive_rate:.2%} for '{reference_group}'. "
            f"DIR of {dir_value:.3f} indicates {'no significant disparity' if status == 'PASS' else 'a significant disparity'} in positive outcomes."
        )
        
        results.append(BiasMetricResult(
            metric_name=BiasMetric.DISPARATE_IMPACT,
            sensitive_attribute=sensitive_attr,
            group=str(group),
            reference_group=reference_group,
            value=dir_value,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            status=status,
            interpretation=interpretation
        ))
    return results

# Execute bias checks
bias_results = []
POSITIVE_LABEL_VALUE = 1 # Assuming 1 indicates fraud
# Set a consistent reference group for each sensitive attribute
REFERENCE_GROUPS = {
    'customer_gender': 'Female', # Assume Female as reference
    'customer_age_group': '31-50', # Assume 31-50 as reference
    'customer_region': 'North' # Assume North as reference
}

for attr in SENSITIVE_ATTRIBUTES:
    if attr in REFERENCE_GROUPS:
        ref_group = REFERENCE_GROUPS[attr]
        # DPD
        bias_results.extend(calculate_demographic_parity_difference(training_data, attr, LABEL_COLUMN, POSITIVE_LABEL_VALUE, ref_group))
        # DIR
        bias_results.extend(calculate_disparate_impact_ratio(training_data, attr, LABEL_COLUMN, POSITIVE_LABEL_VALUE, ref_group))
    else:
        print(f"Warning: No reference group defined for sensitive attribute '{attr}'. Skipping bias calculation.")


print("--- Bias Assessment Results ---")
for result in bias_results:
    print(f"[{result.status}] {result.metric_name} for '{result.sensitive_attribute}' (Group: '{result.group}', Ref: '{result.reference_group}'): Value = {result.value:.3f}. {result.interpretation}")

```

### 4.3 Explanation of Execution: Bias Assessment

The bias assessment reveals critical findings that Sarah needs to address. For `customer_gender`, the **Demographic Parity Difference** and **Disparate Impact Ratio** for 'Male' customers compared to the 'Female' reference group likely report **FAIL** statuses (as designed with the introduced bias). This indicates that male customers are disproportionately flagged as fraudulent in the training data. This finding is a high-risk concern for GlobalTrust, potentially leading to unfair treatment and violating anti-discrimination regulations.

Similarly, the `customer_age_group` assessment might show a **FAIL** for the '18-30' group compared to the '31-50' reference, indicating an elevated fraud rate in the training data for younger customers. Sarah recognizes that these biases in the training data, if not mitigated, will be learned and amplified by the *TransactionGuard AI* model, leading to discriminatory outcomes. This necessitates further investigation into the source of this disparity and potential data re-sampling or re-weighting strategies to promote fairness.

---

## 5. Generating Auditable Reports and Evidence Manifest

Having performed the detailed quality and bias checks, Sarah's next crucial step is to consolidate all findings into structured, auditable reports. This is not just about summarizing data; it's about creating verifiable artifacts that can be presented to internal governance committees and external regulators. She also generates an `EvidenceManifest` to ensure the entire audit process, from input data to output reports, is traceable and tamper-proof.

### 5.1 Story + Context + Real-World Relevance

For GlobalTrust Financial, every AI system deployment requires a clear audit trail. Sarah's reports serve as the official record of the pre-deployment audit. The `data_quality_report.json` and `bias_metrics.json` provide granular details, while the `evidence_manifest.json` acts as a cryptographic fingerprint of the entire audit, linking findings to the specific input data and tools used. This level of detail is paramount for demonstrating compliance and accountability to stakeholders and regulators.

The `inputs_hash` is calculated by hashing the entire raw input dataset. The `outputs_hash` is a dictionary of SHA256 hashes for each generated report file.
$$ \text{SHA256 Hash} = \text{hashlib.sha256}(\text{data.encode()}).\text{hexdigest}() $$

### 5.2 Code Cell (Function Definition + Function Execution)

```python
OUTPUT_DIR = "audit_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_sha256_hash(data: str) -> str:
    """Calculates the SHA256 hash of a string."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def generate_data_quality_report_json(findings: List[DataQualityFinding], output_path: str) -> str:
    """Generates a JSON report for data quality findings."""
    report_content = [f.model_dump() for f in findings]
    with open(output_path, 'w') as f:
        json.dump(report_content, f, indent=4)
    print(f"Data Quality Report saved to {output_path}")
    return calculate_sha256_hash(json.dumps(report_content, indent=4))

def generate_bias_metrics_report_json(results: List[BiasMetricResult], output_path: str) -> str:
    """Generates a JSON report for bias metric results."""
    report_content = [r.model_dump() for r in results]
    with open(output_path, 'w') as f:
        json.dump(report_content, f, indent=4)
    print(f"Bias Metrics Report saved to {output_path}")
    return calculate_sha256_hash(json.dumps(report_content, indent=4))

def generate_evidence_manifest_json(
    df: pd.DataFrame, 
    dq_findings_hash: str, 
    bias_results_hash: str, 
    dq_report_filename: str,
    bias_report_filename: str,
    output_path: str,
    team_or_user: str = "Sarah Chen (Data Risk Lead)",
    app_version: str = "1.0.0"
) -> str:
    """Generates the Evidence Manifest JSON."""
    
    # Calculate input data hash (e.g., hash of the dataframe's string representation or a key column hash)
    # For a robust audit, one might hash the original CSV. Here, we hash the current state of the DF.
    input_data_string = df.to_json(orient='records', date_format='iso') # Using JSON representation for hashing
    inputs_hash = calculate_sha256_hash(input_data_string)

    output_hashes = {
        dq_report_filename: dq_findings_hash,
        bias_report_filename: bias_results_hash,
    }

    artifacts = [
        Artifact(file_name=dq_report_filename, file_hash=dq_findings_hash, description="Detailed data quality findings."),
        Artifact(file_name=bias_report_filename, file_hash=bias_results_hash, description="Detailed bias assessment results.")
    ]
    
    manifest = EvidenceManifest(
        team_or_user=team_or_user,
        app_version=app_version,
        inputs_hash=inputs_hash,
        outputs_hash=output_hashes,
        artifacts=artifacts
    )
    
    manifest_content = manifest.model_dump_json(indent=4)
    with open(output_path, 'w') as f:
        f.write(manifest_content)
    print(f"Evidence Manifest saved to {output_path}")
    return calculate_sha256_hash(manifest_content)

# Define output filenames
DATA_QUALITY_REPORT_FILENAME = "data_quality_report.json"
BIAS_METRICS_REPORT_FILENAME = "bias_metrics.json"
EVIDENCE_MANIFEST_FILENAME = "evidence_manifest.json"

# Generate reports and capture their hashes
dq_report_hash = generate_data_quality_report_json(dq_findings, os.path.join(OUTPUT_DIR, DATA_QUALITY_REPORT_FILENAME))
bias_report_hash = generate_bias_metrics_report_json(bias_results, os.path.join(OUTPUT_DIR, BIAS_METRICS_REPORT_FILENAME))

# Generate evidence manifest
evidence_manifest_hash = generate_evidence_manifest_json(
    df=training_data,
    dq_findings_hash=dq_report_hash,
    bias_results_hash=bias_report_hash,
    dq_report_filename=DATA_QUALITY_REPORT_FILENAME,
    bias_report_filename=BIAS_METRICS_REPORT_FILENAME,
    output_path=os.path.join(OUTPUT_DIR, EVIDENCE_MANIFEST_FILENAME)
)

print(f"\nAll audit artifacts generated in '{OUTPUT_DIR}' directory.")
print(f"Data Quality Report Hash: {dq_report_hash}")
print(f"Bias Metrics Report Hash: {bias_report_hash}")
print(f"Evidence Manifest Hash: {evidence_manifest_hash}")

```

### 5.3 Explanation of Execution: Reports and Evidence Manifest

Sarah now has three critical files in her `audit_artifacts` directory: `data_quality_report.json`, `bias_metrics.json`, and `evidence_manifest.json`. The JSON reports provide detailed, machine-readable summaries of all individual findings, including metric values, thresholds, and PASS/FAIL statuses. This granular detail is invaluable for developers and data engineers who will implement the mitigation strategies.

Crucially, the `evidence_manifest.json` binds the entire audit together. It contains hashes of the input training data and each generated report. This cryptographic linkage ensures that:
1.  **Provenance**: Any future review can verify exactly which dataset was audited.
2.  **Integrity**: It can be proven that the reports have not been tampered with since their generation.

For Sarah, this output is the backbone of her audit trail, allowing GlobalTrust to demonstrate robust governance and compliance with regulatory requirements, providing confidence in the audit process itself.

---

## 6. Executive Summary and Mitigation Recommendations

The final and most critical step for Sarah is to synthesize all audit findings into a concise executive summary for the AI Governance Committee. This document must clearly articulate the key risks identified, provide a summary PASS/FAIL status, and propose actionable, testable mitigation recommendations. This is where technical findings are translated into strategic insights for leadership.

### 6.1 Story + Context + Real-World Relevance

Sarah understands that the AI Governance Committee needs a high-level overview, not raw data. Her executive summary must highlight the most significant data quality and bias risks, explaining their potential impact on GlobalTrust and its customers. More importantly, she must provide concrete recommendations that her team or data engineering can implement to resolve these issues before the *TransactionGuard AI* model goes live. This direct connection from problem to solution demonstrates proactive risk management.

### 6.2 Code Cell (Function Definition + Function Execution)

```python
def generate_executive_summary_md(
    dataset_metadata: DatasetMetadata,
    dq_findings: List[DataQualityFinding],
    bias_results: List[BiasMetricResult],
    output_path: str
) -> str:
    """Generates a markdown executive summary of the audit findings."""

    summary_content = []
    
    summary_content.append("# Pre-Deployment AI System Risk Audit: Executive Summary")
    summary_content.append(f"## TransactionGuard AI Model Training Data")
    summary_content.append(f"**Audit Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    summary_content.append(f"**Auditor:** {dataset_metadata.team_or_user if hasattr(dataset_metadata, 'team_or_user') else 'Sarah Chen (Data Risk Lead)'}")
    summary_content.append(f"**Dataset ID:** {dataset_metadata.dataset_id}")
    summary_content.append(f"**Source System:** {dataset_metadata.source_system}, **Ingestion Date:** {dataset_metadata.ingestion_date.strftime('%Y-%m-%d')}")
    summary_content.append(f"**Label Column:** `{dataset_metadata.label_column}`")
    summary_content.append(f"**Sensitive Attributes:** {', '.join([f'`{s_attr}`' for s_attr in dataset_metadata.sensitive_attributes])}")
    summary_content.append("\n---")

    summary_content.append("## 1. Overall Audit Status")
    
    # Determine overall status
    overall_status = "PASS"
    risk_summary = []

    # Data Quality Summary
    dq_failed = [f for f in dq_findings if f.status == "FAIL"]
    if dq_failed:
        overall_status = "FAIL"
        risk_summary.append("### Data Quality Findings (FAIL)")
        for f in dq_failed:
            summary_content.append(f"- **{f.issue_type.value}** on `{f.column if f.column else 'Dataset'}`: {f.description}")
            if f.issue_type == DataQualityIssue.MISSINGNESS and f.column == 'customer_age':
                risk_summary.append(f"  - High **MISSINGNESS** in `customer_age` ({f.metric_value:.2%}) affecting data completeness and potential model bias.")
            elif f.issue_type == DataQualityIssue.DUPLICATES:
                risk_summary.append(f"  - Significant **DUPLICATE** rows ({f.metric_value:.2%}) potentially skewing training data.")
            elif f.issue_type == DataQualityIssue.OUTLIERS and f.column == 'amount':
                risk_summary.append(f"  - High rate of **OUTLIERS** in `amount` ({f.metric_value:.2%}) could distort model learning for transaction values.")

    # Bias Metrics Summary
    bias_failed = [b for b in bias_results if b.status == "FAIL"]
    if bias_failed:
        overall_status = "FAIL"
        summary_content.append("\n### Bias Metrics Findings (FAIL)")
        for b in bias_failed:
            summary_content.append(f"- **{b.metric_name.value}** for `{b.sensitive_attribute}` group `{b.group}` (vs. `{b.reference_group}`): Value {b.value:.3f}. {b.interpretation}")
            risk_summary.append(f"  - Significant **BIAS** detected in `{b.sensitive_attribute}` for group `{b.group}` (Metric: {b.metric_name.value}). Potential for unfair outcomes.")

    summary_content.append(f"\n**Overall Audit Verdict:** **{'FAIL' if overall_status == 'FAIL' else 'PASS'}**")
    
    summary_content.append("\n## 2. Key Risks Identified")
    if risk_summary:
        for risk in risk_summary:
            summary_content.append(risk)
    else:
        summary_content.append("- No critical risks identified. Dataset appears robust.")

    summary_content.append("\n## 3. Mitigation Recommendations")
    summary_content.append("Based on the audit findings, the following mitigation actions are recommended:")

    if any(f.issue_type == DataQualityIssue.MISSINGNESS and f.status == "FAIL" for f in dq_findings):
        summary_content.append("- **Missing Values:** Implement a robust imputation strategy (e.g., median imputation, K-nearest neighbors) for `customer_age` to address high missingness. Investigate upstream data collection processes for completeness.")
    if any(f.issue_type == DataQualityIssue.DUPLICATES and f.status == "FAIL" for f in dq_findings):
        summary_content.append("- **Duplicate Rows:** De-duplicate the training dataset to ensure each transaction record is unique. Review data ingestion pipelines to prevent future duplicate entries.")
    if any(f.issue_type == DataQualityIssue.OUTLIERS and f.status == "FAIL" for f in dq_findings):
        summary_content.append("- **Outliers in `amount`:** Investigate the nature of high-value `amount` outliers. Consider Winsorization or robust scaling techniques during feature engineering if they represent legitimate but rare events, or flag for data cleansing if they are errors.")
    if any(b.status == "FAIL" for b in bias_results):
        summary_content.append("- **Bias in Sensitive Attributes:**")
        for b in [br for br in bias_results if br.status == "FAIL"]:
            summary_content.append(f"  - For `{b.sensitive_attribute}` (Group: `{b.group}`), address the `{b.metric_name.value}` disparity. Consider data re-sampling (e.g., oversampling under-represented positive outcomes, or undersampling over-represented negative outcomes), re-weighting, or exploring fairness-aware learning algorithms.")
            summary_content.append(f"  - Conduct a root-cause analysis to understand why these disparities exist in the data (e.g., historical data collection practices, inherent socio-economic factors) to inform long-term data strategy.")

    if not dq_failed and not bias_failed:
        summary_content.append("- The audit did not identify any critical data quality or bias issues requiring immediate mitigation. The training data appears suitable for model deployment.")

    summary_content.append("\n---")
    summary_content.append("## 4. Next Steps")
    summary_content.append("- Present this report to the AI Governance Committee for review.")
    summary_content.append("- Initiate work on identified mitigation strategies with the Data Engineering and ML teams.")
    summary_content.append("- Re-audit the dataset post-mitigation to confirm effectiveness.")
    summary_content.append("- Document findings and actions in GlobalTrust Financial's AI Risk Register.")

    executive_summary_content = "\n".join(summary_content)
    with open(output_path, 'w') as f:
        f.write(executive_summary_content)
    print(f"Executive Summary saved to {output_path}")
    return calculate_sha256_hash(executive_summary_content)


# Generate the executive summary
EXECUTIVE_SUMMARY_FILENAME = "case_executive_summary.md"
executive_summary_hash = generate_executive_summary_md(
    dataset_metadata=dataset_metadata,
    dq_findings=dq_findings,
    bias_results=bias_results,
    output_path=os.path.join(OUTPUT_DIR, EXECUTIVE_SUMMARY_FILENAME)
)

# Update evidence manifest with executive summary hash
evidence_manifest_path = os.path.join(OUTPUT_DIR, EVIDENCE_MANIFEST_FILENAME)
with open(evidence_manifest_path, 'r') as f:
    manifest_data = json.load(f)

manifest_data['outputs_hash'][EXECUTIVE_SUMMARY_FILENAME] = executive_summary_hash
manifest_data['artifacts'].append(
    Artifact(file_name=EXECUTIVE_SUMMARY_FILENAME, file_hash=executive_summary_hash, description="Consolidated executive summary of audit findings and recommendations.").model_dump()
)

with open(evidence_manifest_path, 'w') as f:
    json.dump(manifest_data, f, indent=4)

print(f"\nExecutive Summary Hash: {executive_summary_hash}")
print(f"Evidence Manifest updated with Executive Summary hash.")

```

### 6.3 Explanation of Execution: Executive Summary

The `case_executive_summary.md` file now provides a clear, high-level overview of the audit. For Sarah, this document is her primary communication tool with the AI Governance Committee. It summarizes the **FAIL** statuses for missing values, duplicates, outliers, and most critically, the detected bias in `customer_gender` and `customer_age_group`. Each identified risk is accompanied by concrete, testable mitigation recommendations, such as robust imputation for missing `customer_age` and re-sampling strategies to address the identified biases.

By providing these actionable recommendations, Sarah is not just identifying problems; she is providing a pathway for remediation. This empowers the Committee to make an informed decision on the *TransactionGuard AI* model's deployment, knowing that a thorough audit has been conducted and a clear plan is in place to address the identified risks, upholding GlobalTrust's commitment to responsible AI. The updated `evidence_manifest.json` ensures that even this final report is fully auditable.
