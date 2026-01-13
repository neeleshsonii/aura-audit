from fairlearn.metrics import (
    selection_rate, 
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def audit_for_bias(y_true, y_pred, sensitive_features):
    """
    Task 3.1: Calculate fairness metrics using Fairlearn.
    - Measure Demographic Parity Difference.
    - Measure Selection Rate across groups.
    """
    print("\n" + "="*60)
    print("BIAS AUDIT RESULTS")
    print("="*61)
    # Convert sensitive features to DataFrame if needed
    if isinstance(sensitive_features, np.ndarray):
        sf_df = pd.DataFrame({'sensitive_feature': sensitive_features})
    elif isinstance(sensitive_features, pd.Series):
        sf_df = pd.DataFrame({'sensitive_feature': sensitive_features.values})
    else:
        sf_df = sensitive_features
    
    # Calculate overall metrics
    from sklearn.metrics import accuracy_score
    overall_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    
    # Calculate selection rates per group
    print("\nSelection Rates by Group: ")
    for group in sf_df['sensitive_feature'].unique():
        mask = sf_df['sensitive_feature'] == group
        rate = selection_rate(y_true[mask], y_pred[mask])
        print(f"Group '{group}': {rate:.4f}")
    # Calculate Demographic Parity Difference
    dpd = demographic_parity_difference(
        y_true, 
        y_pred, 
        sensitive_features=sf_df['sensitive_feature']
    )
    print(f"\nDemographic Parity Difference: {dpd:.4f}")
    print("(Closer to 0 is better, threshold typically < 0.1)")
    
    # Calculate Equalized Odds Difference
    eod = equalized_odds_difference(
        y_true,
        y_pred,
        sensitive_features=sf_df['sensitive_feature']
    )
    print(f"Equalized Odds Difference: {eod:.4f}")
    print("(Measures difference in TPR and FPR across groups)")
        
    metrics = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sf_df['sensitive_feature']
    )
    
    print("\nPerformance Metrics by Group: ")
    print(metric_frame.by_group)
    
    # Check for bias flags
    bias_detected = False
    if abs(dpd) > 0.1:
        print("\nWARNING: Demographic Parity Difference exceeds threshold!")
        bias_detected = True
    if abs(eod) > 0.1:
        print("WARNING: Equalized Odds Difference exceeds threshold!")
        bias_detected = True
    
    if not bias_detected:
        print("\n✓ No significant bias detected in fairness metrics.")
    return {
        'demographic_parity_diff': dpd,
        'equalized_odds_diff': eod,
        'metric_frame': metric_frame,
        'bias_detected': bias_detected
    }



def explain_decisions(model, X_sample, feature_names=None, max_display=20):
    """
    Task 3.1: Use SHAP to explain model predictions.
    - Generate summary_plot.
    """
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("="*60)
    
    # Limit sample size for computational efficiency
    if len(X_sample) > 100:
        sample_indices = np.random.RandomState(RANDOM_SEED).choice(
            len(X_sample), 100, replace=False
        )
        X_sample = X_sample[sample_indices]
    
    # Create SHAP explainer
    print("\nGenerating SHAP explanations...")
    print("(This may take a moment...)")
    
    try:
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # For multi-class, shap_values is a list
        if isinstance(shap_values, list):
            shap_values_display = shap_values[0]  # Use first class for visualization
        else:
            shap_values_display = shap_values
        # Generate summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_display, 
            X_sample,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/shap_summary.png', dpi=150, bbox_inches='tight')
        print("SHAP summary plot saved to outputs/shap_summary.png")
        plt.close()
        # Generate waterfall plot for first prediction
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_display[0],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[0],
                data=X_sample[0],
                feature_names=feature_names
            ),
            max_display=15,
            show=False
        )
        plt.savefig('outputs/shap_waterfall.png', dpi=150, bbox_inches='tight')
        print("SHAP waterfall plot saved to outputs/shap_waterfall.png")
        plt.close()
        
        print("\nTop 10 Most Important Features (by mean |SHAP value|):")
        mean_abs_shap = np.abs(shap_values_display).mean(axis=0)
        if feature_names:
            feature_importance = sorted(
                zip(feature_names, mean_abs_shap),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for i, (feat, importance) in enumerate(feature_importance, 1):
                print(f"{i}. {feat}: {importance:.4f}")
        return shap_values, explainer
        
    except Exception as e:
        print(f"Note: SHAP analysis encountered an issue: {e}")
        print("Continuing with alternative explanation method...")
        return None, None
    
    
def apply_in_processing_mitigation(X, y, sensitive_features, base_estimator=None):
    """
    Task 2.3: Apply in-processing bias mitigation using Fairlearn.
    - Use ExponentiatedGradient with DemographicParity constraint.
    - Train a fair model that reduces bias during training.
    """
    print("\n" + "="*60)
    print("IN-PROCESSING BIAS MITIGATION")
    print("="*61)
    
    if base_estimator is None:
        base_estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=RANDOM_SEED
        )
    
    print("\nApplying Exponentiated Gradient with Demographic Parity constraint...")
    
    # Create mitigator
    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=DemographicParity(),
        eps=0.01,
        max_iter=50,
        nu=1e-6
    )
    # Train fair model
    mitigator.fit(X, y, sensitive_features=sensitive_features)
    print("Fair model trained successfully")
    print(f"Number of predictors generated: {len(mitigator.predictors_)}")
    return mitigator


def apply_post_processing_mitigation(y_pred, y_pred_proba, sensitive_features, threshold=0.5):
    """
    Task 3.1: Implement a Post-processing guardrail.
    - Adjust thresholds to equalize odds or selection rates.
    """
    print("\n" + "="*60)
    print("POST-PROCESSING BIAS MITIGATION")
    print("="*61)
    
    # Convert to DataFrame
    if isinstance(sensitive_features, np.ndarray):
        sf_df = pd.DataFrame({'sensitive_feature': sensitive_features})
    else:
        sf_df = sensitive_features
    
    # Calculate group-specific thresholds
    adjusted_predictions = y_pred.copy()
    
    print("\nAdjusting decision thresholds per group to improve fairness...")
    # Calculate current selection rates
    for group in sf_df['sensitive_feature'].unique():
        mask = sf_df['sensitive_feature'].values == group
        current_rate = y_pred[mask].mean()
        print(f"Group '{group}' - Current selection rate: {current_rate:.4f}")
    # Calculate overall selection rate as target
    target_rate = y_pred.mean()
    print(f"\nTarget selection rate (overall): {target_rate:.4f}")
    
    # Adjust thresholds per group
    for group in sf_df['sensitive_feature'].unique():
        mask = sf_df['sensitive_feature'].values == group
        group_probas = y_pred_proba[mask]
        
        # Find threshold that achieves target rate
        sorted_probas = np.sort(group_probas)[::-1]
        n_positive = int(len(sorted_probas) * target_rate)
        
        if n_positive > 0 and n_positive < len(sorted_probas):
            adjusted_threshold = sorted_probas[n_positive]
            adjusted_predictions[mask] = (group_probas >= adjusted_threshold).astype(int)
        
        new_rate = adjusted_predictions[mask].mean()
        print(f"Group '{group}' - Adjusted selection rate: {new_rate:.4f}")
    
    print("\nPost-processing adjustment complete")
    
    return adjusted_predictions

def generate_bias_report(audit_results, df, output_path='outputs/bias_audit_report.txt'):
    """Generate a comprehensive bias audit report."""
    os.makedirs('outputs', exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BIAS AUDIT REPORT\n")
        f.write("Aura-Audit Customer Support AI System\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. FAIRNESS METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Demographic Parity Difference: {audit_results['demographic_parity_diff']:.4f}\n")
        f.write(f"Equalized Odds Difference: {audit_results['equalized_odds_diff']:.4f}\n\n")
        
        f.write("2. BIAS DETECTION\n")
        f.write("-"*70 + "\n")
        if audit_results['bias_detected']:
            f.write("⚠️  BIAS DETECTED: The model shows unfair treatment across groups.\n")
            f.write("Recommendation: Apply bias mitigation techniques.\n\n")
        else:
            f.write("No significant bias detected.\n\n")
        
        f.write("3. GROUP-LEVEL PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(str(audit_results['metric_frame'].by_group))
        f.write("\n\n")
        
        f.write("4. RECOMMENDATIONS\n")
        f.write("-"*70 + "\n")
        f.write("- Regular monitoring of fairness metrics\n")
        f.write("- Diverse training data collection\n")
        f.write("- Stakeholder feedback integration\n")
        f.write("- Periodic model retraining with fairness constraints\n")
    
    print(f"\nBias audit report saved to {output_path}")

if __name__ == "__main__":
    print("Running Bias Audit...")
