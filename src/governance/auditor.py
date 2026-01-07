from fairlearn.metrics import selection_rate, demographic_parity_difference
import shap
import pandas as pd

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def audit_for_bias(y_true, y_pred, sensitive_features):
    """
    Task 3.1: Calculate fairness metrics using Fairlearn.
    - Measure Demographic Parity Difference.
    - Measure Selection Rate across groups.
    """
    # TODO: Candidate implements this
    pass

def explain_decisions(model, X_sample):
    """
    Task 3.1: Use SHAP to explain model predictions.
    - Generate summary_plot.
    """
    # TODO: Candidate implements this
    pass

def apply_post_processing_mitigation(y_pred, sensitive_features):
    """
    Task 3.1: Implement a Post-processing guardrail.
    - Adjust thresholds to equalize odds or selection rates.
    """
    # TODO: Candidate implements this
    pass

if __name__ == "__main__":
    print("Running Bias Audit...")
