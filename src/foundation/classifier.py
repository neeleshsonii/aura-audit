import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def run_supervised_baseline(X, y):
    """
    Task 1.2: Train a Supervised Random Forest.
    - Split data (test_size=0.2, random_state=RANDOM_SEED)
    - Train RandomForestClassifier(random_state=RANDOM_SEED)
    - Return accuracy and report
    """
    # TODO: Candidate implements this
    pass

def run_semi_supervised_learning(X_unlabelled, X_labelled, y_labelled):
    """
    Task 1.3: Use LabelSpreading to propagate labels to the unlabelled set.
    - Combine labelled and unlabelled data
    - Set labels for unlabelled as -1
    - Fit LabelSpreading
    - Return the full pseudo-labelled dataset
    """
    # TODO: Candidate implements this
    pass

if __name__ == "__main__":
    # Load data
    print("Running classification...")
