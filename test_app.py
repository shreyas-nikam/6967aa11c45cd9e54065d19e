
from streamlit.testing.v1 import AppTest
import pandas as pd
import os
import json
import numpy as np

# Helper function to perform data loading and metadata capture,
# which is a prerequisite for subsequent pages.
def _setup_intro_page_with_data_and_metadata(at: AppTest) -> AppTest:
    """
    Navigates to the Introduction page, loads synthetic demo data,
    and processes the data to capture metadata.
    """
    at.selectbox(label="Go to")[0].set_value("Introduction & Data Upload").run()
    at.button(label="Load Synthetic Demo Data")[0].click().run()
    
    # After loading data, the app reruns, and new widgets become available.
    at.run() 
    
    # Assume 'is_fraud' is the default label column and 'customer_gender' etc. are sensitive attributes.
    # The app code sets these by default if they exist in the synthetic data,
    # so we don't need to explicitly interact with the selectboxes unless we want to change defaults.
    
    # Click "Process Data & Capture Metadata" button.
    # The index of this button might vary depending on previous interactions,
    # so we search by label for robustness.
    process_button_label = "Process Data & Capture Metadata"
    process_button_found = False
    for i, button_widget in enumerate(at.button):
        if button_widget.label == process_button_label:
            at.button[i].click().run()
            process_button_found = True
            break
            
    if not process_button_found:
        raise ValueError(f"Button '{process_button_label}' not found after loading demo data.")
    
    # Rerun the app to reflect state changes after processing metadata
    at.run()
    return at

# Helper function to complete the Data Quality Audit section.
def _setup_dq_audit_page_complete(at: AppTest) -> AppTest:
    """
    Completes the Introduction page setup and runs Data Quality checks.
    """
    at = _setup_intro_page_with_data_and_metadata(at)
    at.selectbox(label="Go to")[0].set_value("Data Quality Audit").run()
    
    # Click "Run All Data Quality Checks". This is the only direct action button on this page.
    at.button(label="Run All Data Quality Checks")[0].click().run()
    
    # Rerun the app to reflect state changes after DQ checks
    at.run()
    return at

# Helper function to complete the Bias Assessment section.
def _setup_bias_assessment_page_complete(at: AppTest) -> AppTest:
    """
    Completes Data Quality Audit setup and runs Bias Assessment.
    """
    at = _setup_dq_audit_page_complete(at)
    at.selectbox(label="Go to")[0].set_value("Bias Assessment").run()
    
    # After navigating, the page reruns.
    at.run() 
    
    # Select positive label value (assuming 'is_fraud' has 1 as positive).
    # This is the first radio button on the page.
    if len(at.radio) > 0:
        at.radio[0].set_value(1).run()
    else:
        raise ValueError("No radio button found for positive label selection.")

    # Re-run after setting radio button to ensure reference group selectboxes are active and indexed correctly
    at.run() 

    # Set reference groups for sensitive attributes.
    # Assumes 'customer_gender', 'customer_age_group', 'customer_region' from synthetic data.
    # The selectboxes are dynamically created and appear after the 'Go to' navigation selectbox (index 0).
    # We rely on the order of sensitive attributes defined in the app's default list.
    
    # Selectbox for 'customer_gender' (likely index 1 after navigation)
    if len(at.selectbox) > 1 and at.selectbox[1].label.startswith("Select Reference Group for 'customer_gender'"):
        if 'Female' in at.selectbox[1].options:
            at.selectbox[1].set_value('Female').run()
        else:
            print(f"Warning: 'Female' not found in options for 'customer_gender'. Options: {at.selectbox[1].options}")
    
    # Selectbox for 'customer_age_group' (likely index 2)
    if len(at.selectbox) > 2 and at.selectbox[2].label.startswith("Select Reference Group for 'customer_age_group'"):
        if '31-50' in at.selectbox[2].options:
            at.selectbox[2].set_value('31-50').run()
        else:
            print(f"Warning: '31-50' not found in options for 'customer_age_group'. Options: {at.selectbox[2].options}")
            
    # Selectbox for 'customer_region' (likely index 3)
    if len(at.selectbox) > 3 and at.selectbox[3].label.startswith("Select Reference Group for 'customer_region'"):
        if 'North' in at.selectbox[3].options:
            at.selectbox[3].set_value('North').run()
        else:
            print(f"Warning: 'North' not found in options for 'customer_region'. Options: {at.selectbox[3].options}")

    at.run() # Rerun after setting reference groups

    # Click "Run Bias Assessment" button. Search by label for robustness.
    run_bias_button_label = "Run Bias Assessment"
    bias_button_found = False
    for i, button_widget in enumerate(at.button):
        if button_widget.label == run_bias_button_label:
            at.button[i].click().run()
            bias_button_found = True
            break
            
    if not bias_button_found:
        raise ValueError(f"Button '{run_bias_button_label}' not found on Bias Assessment page.")

    at.run() # Rerun after running bias assessment
    return at

# --- Test Functions ---

def test_initial_page_load():
    """Verify the initial state of the application on load."""
    at = AppTest.from_file("app.py").run()
    assert at.session_state.page == "Introduction & Data Upload"
    assert "QuLab: Data Quality, Provenance & Bias Assessment for Enterprise AI Systems" in at.title[0].value
    assert "Case Study Introduction" in at.markdown[2].value
    assert at.subheader[0].value == "Upload Training Data or Load Demo Data"
    assert at.file_uploader[0].label == "Upload your training dataset (CSV)"
    assert at.button[0].label == "Load Synthetic Demo Data"
    assert at.session_state.df is None
    assert at.session_state.dataset_metadata is None

def test_data_upload_and_metadata_capture():
    """
    Tests loading synthetic demo data, selecting parameters,
    and capturing dataset metadata.
    """
    at = AppTest.from_file("app.py").run()
    
    # Click 'Load Synthetic Demo Data'
    at.button[0].click().run()
    
    assert at.session_state.df is not None
    assert "Synthetic demo data loaded successfully!" in at.success[0].value
    
    # After loading data, app reruns, and new widgets appear.
    at.run() 
    
    # Verify label column and sensitive attributes selectboxes appear
    # at.selectbox[0] is for navigation. So, label column selector is at.selectbox[1].
    # Sensitive attributes is at.multiselect[0].
    assert at.selectbox[1].label == "Select the Label Column (e.g., 'is_fraud')"
    assert at.multiselect[0].label == "Select Sensitive Attributes (e.g., 'customer_gender', 'customer_age_group')"
    
    # Assuming 'is_fraud' is present and selected by default in demo data.
    # Assuming default sensitive attributes are pre-selected in demo data.
    assert at.session_state.label_column == 'is_fraud'
    assert 'customer_gender' in at.session_state.sensitive_attributes
    
    # Click "Process Data & Capture Metadata"
    process_button_label = "Process Data & Capture Metadata"
    process_button_found = False
    for i, button_widget in enumerate(at.button):
        if button_widget.label == process_button_label:
            at.button[i].click().run()
            process_button_found = True
            break
            
    assert process_button_found, f"Button '{process_button_label}' not found."

    assert at.session_state.dataset_metadata is not None
    assert "Dataset metadata captured and schema inferred!" in at.success[1].value
    assert at.json[0].value is not None # Verify JSON output for metadata is present

def test_data_quality_audit_flow():
    """
    Tests navigation to Data Quality Audit page, adjusting thresholds,
    running checks, and verifying results display.
    """
    at = AppTest.from_file("app.py").run()
    at = _setup_intro_page_with_data_and_metadata(at)

    # Navigate to "Data Quality Audit" page
    at.selectbox[0].set_value("Data Quality Audit").run()
    
    # Adjust a threshold to test interaction (e.g., Missing Value Rate Threshold)
    # This is at.number_input[0] on the DQ page.
    at.number_input[0].set_value(0.1).run()

    # Click "Run All Data Quality Checks" button. This is the first button on the DQ page.
    at.button[0].click().run()

    assert at.session_state.dq_findings is not None
    assert len(at.session_state.dq_findings) > 0 # Expect findings after running checks
    assert "Data Quality Checks Complete!" in at.success[0].value
    assert at.dataframe[0].value is not None # Verify findings dataframe is displayed
    assert at.metric[0].label == "Overall Data Quality Status"
    assert at.metric[0].value in ["PASS", "FAIL"] # Status should be displayed

def test_bias_assessment_flow():
    """
    Tests navigation to Bias Assessment page, selecting positive label and reference groups,
    running assessment, and verifying results display.
    """
    at = AppTest.from_file("app.py").run()
    at = _setup_dq_audit_page_complete(at)

    # Navigate to "Bias Assessment" page
    at.selectbox[0].set_value("Bias Assessment").run()
    
    # Select positive label value (e.g., 1 for 'is_fraud'). This is at.radio[0].
    if len(at.radio) > 0:
        at.radio[0].set_value(1).run() 
    else:
        raise ValueError("No radio button found for positive label selection on Bias Assessment page.")
    
    at.run() # Rerun after setting radio button to ensure reference group selectboxes are updated.

    # Set reference groups for sensitive attributes
    # at.selectbox[1] to at.selectbox[3] are typically for 'customer_gender', 'customer_age_group', 'customer_region'
    if len(at.selectbox) > 1 and at.selectbox[1].label.startswith("Select Reference Group for 'customer_gender'"):
        if 'Female' in at.selectbox[1].options: at.selectbox[1].set_value('Female').run()
    if len(at.selectbox) > 2 and at.selectbox[2].label.startswith("Select Reference Group for 'customer_age_group'"):
        if '31-50' in at.selectbox[2].options: at.selectbox[2].set_value('31-50').run()
    if len(at.selectbox) > 3 and at.selectbox[3].label.startswith("Select Reference Group for 'customer_region'"):
        if 'North' in at.selectbox[3].options: at.selectbox[3].set_value('North').run()

    at.run() # Rerun after setting reference groups.

    # Click "Run Bias Assessment" button. Search by label.
    run_bias_button_label = "Run Bias Assessment"
    bias_button_found = False
    for i, button_widget in enumerate(at.button):
        if button_widget.label == run_bias_button_label:
            at.button[i].click().run()
            bias_button_found = True
            break
            
    assert bias_button_found, f"Button '{run_bias_button_label}' not found."

    assert at.session_state.bias_results is not None
    assert len(at.session_state.bias_results) > 0 # Expect results after running assessment
    assert "Bias Assessment Complete!" in at.success[0].value
    assert at.dataframe[0].value is not None # Verify results dataframe is displayed

def test_reports_and_recommendations_flow():
    """
    Tests navigation to Reports & Recommendations, generating all reports,
    and verifying the presence of hash values and download buttons.
    """
    at = AppTest.from_file("app.py").run()
    at = _setup_bias_assessment_page_complete(at) # Complete all previous steps

    # Navigate to "Reports & Recommendations" page
    at.selectbox[0].set_value("Reports & Recommendations").run()
    
    # Click "Generate All Reports and Evidence Manifest". This is the first button on this page.
    at.button[0].click().run()

    assert at.session_state.dq_report_hash is not None
    assert at.session_state.bias_report_hash is not None
    assert at.session_state.executive_summary_hash is not None
    assert at.session_state.evidence_manifest_hash is not None
    
    assert "All reports and evidence manifest generated!" in at.success[0].value
    
    # Verify hash displays using markdown content.
    # The actual index depends on the content and other markdown elements.
    # We check for the presence of the hash string within any markdown.
    assert any(f"**Data Quality Report Hash:** `{at.session_state.dq_report_hash}`" in m.value for m in at.markdown)
    assert any(f"**Bias Metrics Report Hash:** `{at.session_state.bias_report_hash}`" in m.value for m in at.markdown)
    
    # Verify Executive Summary preview and download button
    assert at.subheader[1].value == "Executive Summary (Preview)"
    assert at.download_button[0].label == "Download Executive Summary (Markdown)"
    
    # Verify "Download All Audit Artifacts (ZIP)" button
    assert at.subheader[2].value == "Download All Audit Artifacts"
    assert at.download_button[1].label == "Download All Audit Artifacts (ZIP)"
