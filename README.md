Here's a comprehensive `README.md` file for your Streamlit application lab project, formatted in Markdown:

---

# QuLab: Data Quality, Provenance & Bias Assessment for Enterprise AI Systems

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Description

**QuLab** is a Streamlit-powered application designed to simulate a robust pre-deployment audit workflow for enterprise AI systems, focusing on data quality, provenance, and algorithmic bias. It provides a practical environment for Data Risk Leads to systematically analyze training datasets, identify potential risks, and generate auditable reports before an AI model is deployed in a high-stakes environment.

This application is built around a case study where **Sarah Chen, the Data Risk Lead at GlobalTrust Financial**, must audit the *TransactionGuard AI* fraud detection model's training data. The goal is to ensure the model is fair, reliable, and compliant, preventing issues like unfair outcomes for customers, regulatory fines, and reputational damage.

## Features

QuLab offers a comprehensive suite of tools organized into a guided workflow:

1.  **Introduction & Data Upload**:
    *   **Flexible Data Ingestion**: Upload custom CSV datasets or load synthetic demo data for immediate testing.
    *   **Core Configuration**: Identify the label (target) column and select sensitive attributes for bias assessment.
    *   **Automated Metadata Capture**: Infer data schema, record count, and capture critical provenance details like `source_system` and `ingestion_date` to ensure auditability.

2.  **Data Quality Audit**:
    *   **Missing Value Analysis**: Quantify missing data rates per column and flag columns exceeding configurable thresholds.
    *   **Duplicate Row Detection**: Identify and quantify duplicate records within the dataset, highlighting potential data redundancy and bias sources.
    *   **Outlier Detection (IQR Method)**: Utilize the Interquartile Range (IQR) method to identify extreme values in numerical features, with configurable thresholds and an IQR multiplier.
    *   **Schema Consistency Checks**: Compare the dataset's inferred schema against expected types to detect schema drift, ensuring data compatibility with downstream models.
    *   **Configurable Thresholds**: Adjust tolerance levels for missingness, duplicates, and outliers to align with organizational data governance policies.
    *   **Detailed Findings**: Present a summary of all data quality issues, their severity (PASS/FAIL), and descriptions.

3.  **Bias Assessment**:
    *   **Sensitive Attribute Analysis**: Select and define reference groups for various sensitive attributes (e.g., `customer_gender`, `customer_age_group`).
    *   **Demographic Parity Difference (DPD)**: Calculate DPD to measure the difference in positive outcome rates between protected and reference groups, with configurable min/max thresholds.
        $$ DPD = P(\text{Positive Outcome} | \text{Protected Group}) - P(\text{Positive Outcome} | \text{Reference Group}) $$
    *   **Disparate Impact Ratio (DIR)**: Calculate DIR to measure the ratio of positive outcome rates, with configurable min/max thresholds (e.g., the 80% rule).
        $$ DIR = \frac{P(\text{Positive Outcome} | \text{Protected Group})}{P(\text{Positive Outcome} | \text{Reference Group})} $$
    *   **Bias Metric Results**: Display a clear summary of all calculated bias metrics, including values, statuses (PASS/FAIL), and interpretations.

4.  **Reports & Recommendations**:
    *   **Automated Report Generation**: Generate structured JSON reports for Data Quality Findings and Bias Metrics.
    *   **Executive Summary**: Create a high-level Markdown-formatted executive summary, translating technical findings into strategic insights and actionable mitigation recommendations for leadership.
    *   **Evidence Manifest**: Generate a cryptographic `evidence_manifest.json` file that includes SHA256 hashes of the input data and all generated reports. This ensures the **provenance** and **integrity** of the entire audit trail, critical for regulatory compliance.
    *   **One-Click Download**: Download all generated audit artifacts (JSON reports, Executive Summary, Evidence Manifest) as a single ZIP archive.

## Getting Started

Follow these instructions to set up and run the QuLab application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/quolab-ai-audit.git
    cd quolab-ai-audit
    ```
    *(Note: Replace `your-username` with the actual repository owner's username.)*

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**:
    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies**:
    Create a `requirements.txt` file in the root of your project with the following content:
    ```
    streamlit==1.36.0
    pandas==2.2.2
    numpy==1.26.4
    pydantic==2.8.2
    # Add any other specific versions if needed
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit Application**:
    Ensure your virtual environment is active, then execute:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser (usually `http://localhost:8501`).

2.  **Navigate the Application**:
    *   Use the sidebar navigation to move between the `Introduction & Data Upload`, `Data Quality Audit`, `Bias Assessment`, and `Reports & Recommendations` sections.
    *   **Introduction & Data Upload**: Start by uploading your own CSV data or loading the synthetic demo data. Configure the label column and sensitive attributes.
    *   **Data Quality Audit**: Review and adjust the data quality thresholds, then run the checks.
    *   **Bias Assessment**: Select reference groups for your sensitive attributes and adjust bias thresholds, then run the assessment.
    *   **Reports & Recommendations**: Generate and review the comprehensive audit reports and download all artifacts.

## Project Structure

```
quolab-ai-audit/
├── app.py                      # Main Streamlit application script
├── source.py                   # Contains helper functions, Pydantic models, and core logic for checks and report generation
├── requirements.txt            # Python dependencies
├── audit_artifacts/            # Directory where all generated reports and manifest are saved
│   ├── data_quality_report.json
│   ├── bias_metrics.json
│   ├── case_executive_summary.md
│   └── evidence_manifest.json
└── README.md                   # This file
```

## Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building interactive and user-friendly web applications.
*   **Pandas**: Essential for data manipulation and analysis.
*   **NumPy**: Provides support for large, multi-dimensional arrays and matrices, along with a collection of high-level mathematical functions.
*   **Pydantic**: Used for data validation and settings management, defining the structure of metadata, findings, and results.
*   **`hashlib`**: Standard Python library for secure hashes, used for cryptographic verification of data and reports.
*   **Standard Python Libraries**: `os`, `json`, `zipfile`, `io`, `datetime`, `timezone` for file system operations, JSON handling, archiving, and date/time management.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name` or `bugfix/your-bug-name`).
3.  Make your changes and ensure the code passes any existing tests (or add new ones).
4.  Commit your changes (`git commit -m 'feat: Add new feature'`).
5.  Push to your branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request to the `main` branch of this repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: You should create a `LICENSE` file in your repository if you don't have one already.)*

## Contact

For questions, feedback, or collaborations, please reach out to:

*   **Your Name/Organization**: QuantUniversity
*   **Email**: support@quantuniversity.com
*   **Project Link**: [https://www.quantuniversity.com](https://www.quantuniversity.com)
*   **GitHub**: [https://github.com/quantuniversity](https://github.com/quantuniversity)

---