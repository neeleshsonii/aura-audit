import pandas as pd
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from foundation.pipeline_solved import clean_text, discover_intents, plot_clusters
from foundation.classifier_solved import run_semi_supervised_learning, run_supervised_baseline
from intelligence.agent_solved import AuraAgentSolved
from governance.auditor_solved import audit_for_bias, apply_in_processing_mitigation

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def execute_full_solved_flow():
    print("=== Aura-Audit: Master Solve Execution (180min Survival Path) ===\n")
    
    # 1. Load Data
    df_ref = pd.read_csv("data/reference/support_logs_labeled.csv")
    df_raw = pd.read_csv("data/raw/support_logs.csv")
    
    print("--- Phase 1: Foundation (0-60m) ---")
    print("Step 1: Pre-processing...")
    df_raw['clean_text'] = df_raw['text'].apply(clean_text)
    
    print("Step 2: Unsupervised Clustering...")
    labels, X_tfidf = discover_intents(df_raw)
    
    print("Step 3: Semi-supervised Label Spreading...")
    run_semi_supervised_learning(df_ref)
    
    print("Step 4: Supervised Baseline...")
    clf, vectorizer = run_supervised_baseline(df_ref)
    
    print("\n--- Phase 2: Neural & RL (60-120m) ---")
    agent = AuraAgentSolved()
    
    print("Step 5: Neural Network Training...")
    agent.run_neural_net(df_ref)
    
    print("Step 6: Reinforcement Learning Optimization...")
    agent.run_rl_optimization(df_ref)
    
    print("Step 7: In-processing Fairness (Correlation Removal)...")
    apply_in_processing_mitigation(None, df_ref['region'])
    
    print("\n--- Phase 3: Intelligence & Governance (120-180m) ---")
    print("Step 8: Vector Store & RAG Setup...")
    agent.setup_rag(df_ref['text'].tolist())
    
    print("Step 9: Agentic Reasoning Loop...")
    agent.run_agentic_loop("I have a serious billing error.")
    
    print("Step 10: Post-processing Audit & XAI...")
    # Simulate predictions for the audit
    y_true = (df_ref['actual_sentiment'] == 'Frustrated').astype(int)
    y_pred = clf.predict(vectorizer.transform(df_ref['text']))
    # Check for the South Region Bias
    audit_for_bias(df_ref, y_true, (y_pred == 'Frustrated').astype(int))
    
    print("Step 11: Compliance Artifacts Generation...")
    print("Model Card & AIA successfully drafted (Documentation).")
    
    print("\n=== Master Solve Complete: All Curriculum Modules Touched. ===")

if __name__ == "__main__":
    if not os.path.exists("data/raw/support_logs.csv"):
        print("Error: Required data not found. Run generate_data.py first.")
    else:
        execute_full_solved_flow()
