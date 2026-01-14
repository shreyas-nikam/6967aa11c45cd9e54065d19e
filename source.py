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
    schema: Dict[str, str]  # column: type_as_string
    label_column: str
    sensitive_attributes: List[str]
    record_count: int


class DataQualityFinding(BaseModel):
    issue_type: DataQualityIssue
    column: Optional[str] = None
    metric_value: float
    threshold: float
    status: str  # "PASS" or "FAIL"
    description: str


class BiasMetricResult(BaseModel):
    metric_name: BiasMetric
    sensitive_attribute: str
    group: str
    reference_group: str
    value: float
    threshold_min: float
    threshold_max: float
    status: str  # "PASS" or "FAIL"
    interpretation: str


class Artifact(BaseModel):
    file_name: str
    file_hash: str  # SHA256 hash
    description: str


class EvidenceManifest(BaseModel):
    run_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))
    team_or_user: str = "Sarah Chen (Data Risk Lead)"
    app_version: str = "1.0.0"
    inputs_hash: str  # Hash of the raw input data
    # Hashes of generated artifacts (filename: hash)
    outputs_hash: Dict[str, str]
    artifacts: List[Artifact]


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
        # 5% fraud rate
        'is_fraud': np.random.choice([0, 1], size=num_records, p=[0.95, 0.05])
    }

    df = pd.DataFrame(data)

    # Introduce specific issues for audit demonstration
    # 1. Missing values in customer_age
    missing_indices = np.random.choice(df.index, size=int(
        0.06 * num_records), replace=False)  # 6% missing
    df.loc[missing_indices, 'customer_age'] = np.nan

    # 2. Duplicate rows
    duplicate_indices = np.random.choice(df.index, size=int(
        0.01 * num_records), replace=False)  # 1% duplicates
    df = pd.concat([df, df.loc[duplicate_indices]]).reset_index(drop=True)

    # 3. Outliers in amount
    outlier_indices = np.random.choice(df.index, size=int(
        0.005 * num_records), replace=False)  # 0.5% outliers
    df.loc[outlier_indices, 'amount'] = np.random.uniform(
        low=1000, high=5000, size=len(outlier_indices))

    # 4. Introduce bias: Make fraud slightly more likely for 'Male' and 'Age Group: 18-30'
    # Create age groups first
    bins = [0, 30, 50, 70, 100]
    labels = ['18-30', '31-50', '51-70', '71+']
    df['customer_age_group'] = pd.cut(
        df['customer_age'], bins=bins, labels=labels, right=True, include_lowest=True)
    # Convert to string for consistent handling with sensitive attributes
    df['customer_age_group'] = df['customer_age_group'].astype(str)

    male_18_30_indices = df[(df['customer_gender'] == 'Male') & (
        df['customer_age_group'] == '18-30') & (df['is_fraud'] == 0)].index
    fraud_bias_indices = np.random.choice(male_18_30_indices, size=int(
        0.02 * len(male_18_30_indices)), replace=False)
    # Increase fraud rate for this group
    df.loc[fraud_bias_indices, 'is_fraud'] = 1

    # Finalize dtypes and ensure consistent representation
    df['is_fraud'] = df['is_fraud'].astype(int)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['ingestion_date'] = pd.to_datetime(df['ingestion_date'])

    return df

# Generate the dataset
# training_data = generate_synthetic_fraud_data(num_records=10000)


# Define audit parameters
LABEL_COLUMN = 'is_fraud'
SENSITIVE_ATTRIBUTES = ['customer_gender',
                        'customer_age_group', 'customer_region']
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
# inferred_schema = {col: str(dtype) for col, dtype in training_data.dtypes.items()}
# dataset_metadata = DatasetMetadata(
#     source_system=training_data['source_system'].iloc[0],
#     ingestion_date=training_data['ingestion_date'].iloc[0],
#     schema=inferred_schema,
#     label_column=LABEL_COLUMN,
#     sensitive_attributes=SENSITIVE_ATTRIBUTES,
#     record_count=len(training_data)
# )

# print(f"Dataset loaded with {len(training_data)} records.")
# print("\nFirst 5 rows of the dataset:")
# print(training_data.head())
# print("\nDataset Metadata:")
# print(dataset_metadata.model_dump_json(indent=2))


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


def perform_duplicate_check(df: pd.DataFrame, threshold: float = 0.001) -> DataQualityFinding:
    """
    Performs a duplicate row check on the DataFrame and returns a DataQualityFinding object.
    """
    num_duplicates = df.duplicated().sum()
    total_records = len(df)
    duplicate_rate = num_duplicates / total_records
    status = "PASS" if duplicate_rate <= threshold else "FAIL"
    description = f"Dataset contains {num_duplicates} duplicate rows, which is {duplicate_rate:.2%} of total records."
    if status == "FAIL":
        description += f" This exceeds the threshold of {threshold:.2%}. Potential for biased model training."

    return DataQualityFinding(
        issue_type=DataQualityIssue.DUPLICATES,
        column=None,  # Duplicates apply to the whole dataset, not a single column
        metric_value=duplicate_rate,
        threshold=threshold,
        status=status,
        description=description
    )

# Execute the data quality checks
# dq_findings = []
# dq_findings.extend(perform_missing_value_check(training_data, threshold=0.05)) # 5% missingness threshold
# dq_findings.append(perform_duplicate_check(training_data))

# print("--- Missing Value Check Results ---")
# for finding in [f for f in dq_findings if f.issue_type == DataQualityIssue.MISSINGNESS]:
#     print(f"[{finding.status}] {finding.column}: {finding.description}")

# print("\n--- Duplicate Check Results ---")
# for finding in [f for f in dq_findings if f.issue_type == DataQualityIssue.DUPLICATES]:
#     print(f"[{finding.status}] {finding.description}")


def perform_outlier_check_iqr(df: pd.DataFrame, column: str, threshold_iqr_multiplier: float = 1.5, outlier_threshold: float = 0.01) -> DataQualityFinding:
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
                # Represents a complete mismatch (missing column)
                metric_value=1.0,
                threshold=0.0,  # Any missing column is a fail
                status="FAIL",
                description=f"Column '{col}' expected in schema but not found in dataset."
            ))

    # Check for type mismatches or extra columns
    for col, inferred_type in inferred_schema.items():
        if col not in expected_schema:
            findings.append(DataQualityFinding(
                issue_type=DataQualityIssue.SCHEMA_DRIFT,
                column=col,
                # Represents a complete mismatch (extra column)
                metric_value=1.0,
                threshold=0.0,
                status="FAIL",
                description=f"Column '{col}' found in dataset but not in expected schema (inferred type: {inferred_type})."
            ))
        elif inferred_type != expected_schema[col]:
            findings.append(DataQualityFinding(
                issue_type=DataQualityIssue.SCHEMA_DRIFT,
                column=col,
                metric_value=0.5,  # Represents a type mismatch
                threshold=0.0,  # Any type mismatch is a fail
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
# dq_findings.append(perform_outlier_check_iqr(training_data, 'amount', threshold_iqr_multiplier=1.5))
# dq_findings.extend(perform_schema_check(training_data, EXPECTED_SCHEMA))

# print("--- Outlier Check Results ---")
# for finding in [f for f in dq_findings if f.issue_type == DataQualityIssue.OUTLIERS]:
#     print(f"[{finding.status}] {finding.column}: {finding.description}")

# print("\n--- Schema Consistency Check Results ---")
# for finding in [f for f in dq_findings if f.issue_type == DataQualityIssue.SCHEMA_DRIFT]:
#     print(f"[{finding.status}] {finding.column if finding.column else 'Dataset'}: {finding.description}")

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
            status = "PASS"  # Always passes for reference
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
            status = "PASS"  # Always passes for reference
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

# # Execute bias checks
# bias_results = []
# POSITIVE_LABEL_VALUE = 1 # Assuming 1 indicates fraud
# # Set a consistent reference group for each sensitive attribute
# REFERENCE_GROUPS = {
#     'customer_gender': 'Female', # Assume Female as reference
#     'customer_age_group': '31-50', # Assume 31-50 as reference
#     'customer_region': 'North' # Assume North as reference
# }

# for attr in SENSITIVE_ATTRIBUTES:
#     if attr in REFERENCE_GROUPS:
#         ref_group = REFERENCE_GROUPS[attr]
#         # DPD
#         bias_results.extend(calculate_demographic_parity_difference(training_data, attr, LABEL_COLUMN, POSITIVE_LABEL_VALUE, ref_group))
#         # DIR
#         bias_results.extend(calculate_disparate_impact_ratio(training_data, attr, LABEL_COLUMN, POSITIVE_LABEL_VALUE, ref_group))
#     else:
#         print(f"Warning: No reference group defined for sensitive attribute '{attr}'. Skipping bias calculation.")


# print("--- Bias Assessment Results ---")
# for result in bias_results:
#     print(f"[{result.status}] {result.metric_name} for '{result.sensitive_attribute}' (Group: '{result.group}', Ref: '{result.reference_group}'): Value = {result.value:.3f}. {result.interpretation}")

# OUTPUT_DIR = "audit_artifacts"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    """
    Generates a JSON report for bias metric results.
    """
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
    """
    Generates the Evidence Manifest JSON.
    """

    # Calculate input data hash (e.g., hash of the dataframe's string representation or a key column hash)
    # For a robust audit, one might hash the original CSV. Here, we hash the current state of the DF.
    # Using JSON representation for hashing
    input_data_string = df.to_json(orient='records', date_format='iso')
    inputs_hash = calculate_sha256_hash(input_data_string)

    output_hashes = {
        dq_report_filename: dq_findings_hash,
        bias_report_filename: bias_results_hash,
    }

    artifacts = [
        Artifact(file_name=dq_report_filename, file_hash=dq_findings_hash,
                 description="Detailed data quality findings."),
        Artifact(file_name=bias_report_filename, file_hash=bias_results_hash,
                 description="Detailed bias assessment results.")
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
# dq_report_hash = generate_data_quality_report_json(dq_findings, os.path.join(OUTPUT_DIR, DATA_QUALITY_REPORT_FILENAME))
# bias_report_hash = generate_bias_metrics_report_json(bias_results, os.path.join(OUTPUT_DIR, BIAS_METRICS_REPORT_FILENAME))

# # Generate evidence manifest
# evidence_manifest_hash = generate_evidence_manifest_json(
#     df=training_data,
#     dq_findings_hash=dq_report_hash,
#     bias_results_hash=bias_report_hash,
#     dq_report_filename=DATA_QUALITY_REPORT_FILENAME,
#     bias_report_filename=BIAS_METRICS_REPORT_FILENAME,
#     output_path=os.path.join(OUTPUT_DIR, EVIDENCE_MANIFEST_FILENAME)
# )

# print(f"\nAll audit artifacts generated in '{OUTPUT_DIR}' directory.")
# print(f"Data Quality Report Hash: {dq_report_hash}")
# print(f"Bias Metrics Report Hash: {bias_report_hash}")
# print(f"Evidence Manifest Hash: {evidence_manifest_hash}")


def generate_executive_summary_md(
    dataset_metadata: DatasetMetadata,
    dq_findings: List[DataQualityFinding],
    bias_results: List[BiasMetricResult],
    output_path: str
) -> str:
    """Generates a markdown executive summary of the audit findings."""

    summary_content = []

    summary_content.append(
        "# Pre-Deployment AI System Risk Audit: Executive Summary")
    summary_content.append(f"## TransactionGuard AI Model Training Data")
    summary_content.append(
        f"\n**Audit Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    summary_content.append(
        f"\n**Auditor:** {dataset_metadata.team_or_user if hasattr(dataset_metadata, 'team_or_user') else 'Sarah Chen (Data Risk Lead)'}")
    summary_content.append(f"\n**Dataset ID:** {dataset_metadata.dataset_id}")
    summary_content.append(
        f"\n**Source System:** {dataset_metadata.source_system}, **Ingestion Date:** {dataset_metadata.ingestion_date.strftime('%Y-%m-%d')}")
    summary_content.append(
        f"\n**Label Column:** `{dataset_metadata.label_column}`")
    summary_content.append(
        f"\n**Sensitive Attributes:** {', '.join([f'`{s_attr}`' for s_attr in dataset_metadata.sensitive_attributes])}")
    summary_content.append("\n---")

    summary_content.append("## 1. Overall Audit Status")

    # Determine overall status
    overall_status = "PASS"
    risk_summary = []

    # Data Quality Summary
    dq_failed = [f for f in dq_findings if f.status == "FAIL"]
    if dq_failed:
        overall_status = "FAIL"
        summary_content.append("\n### Data Quality Findings (FAIL)")
        for f in dq_failed:
            summary_content.append(
                f"- **{f.issue_type.value}** on `{f.column if f.column else 'Dataset'}`: {f.description}")
            # Dynamically add to risk summary based on actual findings
            if f.issue_type == DataQualityIssue.MISSINGNESS:
                risk_summary.append(
                    f"  - High **MISSINGNESS** in `{f.column}` ({f.metric_value:.2%}) affecting data completeness and potential model bias.")
            elif f.issue_type == DataQualityIssue.DUPLICATES:
                risk_summary.append(
                    f"  - Significant **DUPLICATE** rows ({f.metric_value:.2%}) potentially skewing training data.")
            elif f.issue_type == DataQualityIssue.OUTLIERS:
                risk_summary.append(
                    f"  - High rate of **OUTLIERS** in `{f.column}` ({f.metric_value:.2%}) could distort model learning.")
            elif f.issue_type == DataQualityIssue.SCHEMA_DRIFT:
                risk_summary.append(
                    f"  - **SCHEMA MISMATCH** in `{f.column}`: {f.description}")

    # Bias Metrics Summary
    bias_failed = [b for b in bias_results if b.status == "FAIL"]
    if bias_failed:
        overall_status = "FAIL"
        summary_content.append("\n### Bias Metrics Findings (FAIL)")
        for b in bias_failed:
            summary_content.append(
                f"- **{b.metric_name.value}** for `{b.sensitive_attribute}` group `{b.group}` (vs. `{b.reference_group}`): Value {b.value:.3f}. {b.interpretation}")
            risk_summary.append(
                f"  - Significant **BIAS** detected in `{b.sensitive_attribute}` for group `{b.group}` (Metric: {b.metric_name.value}). Potential for unfair outcomes.")

    summary_content.append(
        f"\n**Overall Audit Verdict:** **{'FAIL' if overall_status == 'FAIL' else 'PASS'}**")

    summary_content.append("\n## 2. Key Risks Identified")
    if risk_summary:
        for risk in risk_summary:
            summary_content.append(risk)
    else:
        summary_content.append(
            "- No critical risks identified. Dataset appears robust.")

    summary_content.append("\n## 3. Mitigation Recommendations")
    summary_content.append(
        "Based on the audit findings, the following mitigation actions are recommended:")

    # Dynamic recommendations based on actual findings
    missing_fails = [f for f in dq_findings if f.issue_type ==
                     DataQualityIssue.MISSINGNESS and f.status == "FAIL"]
    if missing_fails:
        cols = ', '.join([f"`{f.column}`" for f in missing_fails])
        summary_content.append(
            f"- **Missing Values:** Implement a robust imputation strategy (e.g., median imputation, K-nearest neighbors) for {cols} to address high missingness. Investigate upstream data collection processes for completeness.")

    if any(f.issue_type == DataQualityIssue.DUPLICATES and f.status == "FAIL" for f in dq_findings):
        summary_content.append(
            "- **Duplicate Rows:** De-duplicate the training dataset to ensure each transaction record is unique. Review data ingestion pipelines to prevent future duplicate entries.")

    outlier_fails = [f for f in dq_findings if f.issue_type ==
                     DataQualityIssue.OUTLIERS and f.status == "FAIL"]
    if outlier_fails:
        cols = ', '.join([f"`{f.column}`" for f in outlier_fails[:3]])
        if len(outlier_fails) > 3:
            cols += f", and {len(outlier_fails) - 3} more"
        summary_content.append(
            f"- **Outliers in {cols}:** Investigate the nature of high-value outliers. Consider Winsorization or robust scaling techniques during feature engineering if they represent legitimate but rare events, or flag for data cleansing if they are errors.")

    schema_fails = [f for f in dq_findings if f.issue_type ==
                    DataQualityIssue.SCHEMA_DRIFT and f.status == "FAIL"]
    if schema_fails:
        cols = ', '.join([f"`{f.column}`" for f in schema_fails])
        summary_content.append(
            f"- **Schema Mismatches in {cols}:** Correct data type inconsistencies to prevent pipeline errors. Implement schema validation checks in the data ingestion process.")

    if any(b.status == "FAIL" for b in bias_results):
        summary_content.append("- **Bias in Sensitive Attributes:**")
        for b in [br for br in bias_results if br.status == "FAIL"]:
            summary_content.append(f"  - For `{b.sensitive_attribute}` (Group: `{b.group}`), address the `{b.metric_name.value}` disparity. Consider data re-sampling (e.g., oversampling under-represented positive outcomes, or undersampling over-represented negative outcomes), re-weighting, or exploring fairness-aware learning algorithms.")
            summary_content.append(
                f"  - Conduct a root-cause analysis to understand why these disparities exist in the data (e.g., historical data collection practices, inherent socio-economic factors) to inform long-term data strategy.")

    if not dq_failed and not bias_failed:
        summary_content.append(
            "- The audit did not identify any critical data quality or bias issues requiring immediate mitigation. The training data appears suitable for model deployment.")

    summary_content.append("\n---")
    summary_content.append("## 4. Next Steps")
    summary_content.append(
        "- Present this report to the AI Governance Committee for review.")
    summary_content.append(
        "- Initiate work on identified mitigation strategies with the Data Engineering and ML teams.")
    summary_content.append(
        "- Re-audit the dataset post-mitigation to confirm effectiveness.")
    summary_content.append(
        "- Document findings and actions in GlobalTrust Financial's AI Risk Register.")

    executive_summary_content = "\n".join(summary_content)
    with open(output_path, 'w') as f:
        f.write(executive_summary_content)
    print(f"Executive Summary saved to {output_path}")
    return calculate_sha256_hash(executive_summary_content)


# Generate the executive summary
EXECUTIVE_SUMMARY_FILENAME = "case_executive_summary.md"
# executive_summary_hash = generate_executive_summary_md(
#     dataset_metadata=dataset_metadata,
#     dq_findings=dq_findings,
#     bias_results=bias_results,
#     output_path=os.path.join(OUTPUT_DIR, EXECUTIVE_SUMMARY_FILENAME)
# )

# # Update evidence manifest with executive summary hash
# evidence_manifest_path = os.path.join(OUTPUT_DIR, EVIDENCE_MANIFEST_FILENAME)
# with open(evidence_manifest_path, 'r') as f:
#     manifest_data = json.load(f)

# manifest_data['outputs_hash'][EXECUTIVE_SUMMARY_FILENAME] = executive_summary_hash
# manifest_data['artifacts'].append(
#     Artifact(file_name=EXECUTIVE_SUMMARY_FILENAME, file_hash=executive_summary_hash, description="Consolidated executive summary of audit findings and recommendations.").model_dump()
# )

# with open(evidence_manifest_path, 'w') as f:
#     json.dump(manifest_data, f, indent=4)

# print(f"\nExecutive Summary Hash: {executive_summary_hash}")
# print(f"Evidence Manifest updated with Executive Summary hash.")
