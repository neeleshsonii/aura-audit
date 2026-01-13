import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def run_supervised_baseline(X, y):
    """
    Task 1.2: Train a Supervised Random Forest.
    - Split data (test_size=0.2, random_state=RANDOM_SEED)
    - Train RandomForestClassifier(random_state=RANDOM_SEED)
    - Return accuracy and report
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1
    )
    rf_classifier.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("\nRandom Forest Baseline Results: ")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report: ")
    print(report)
    return rf_classifier, accuracy, report, X_test, y_test, y_pred

def run_semi_supervised_learning(X_unlabelled, X_labelled, y_labelled):
    """
    Task 1.3: Use LabelSpreading to propagate labels to the unlabelled set.
    - Combine labelled and unlabelled data
    - Set labels for unlabelled as -1
    - Fit LabelSpreading
    - Return the full pseudo-labelled dataset
    """
    # Combine labelled and unlabelled data
    X_combined = np.vstack([X_labelled, X_unlabelled])
    # Create labels: -1 for unlabelled data
    y_combined = np.concatenate([
        y_labelled,
        np.full(X_unlabelled.shape[0], -1)
    ])
    # Apply Label Spreading
    label_spreading = LabelSpreading(
        kernel='knn',
        n_neighbors=7,
        max_iter=30,
        alpha=0.2
    )
    
    label_spreading.fit(X_combined, y_combined)
    # Get pseudo labels for all data
    pseudo_labels = label_spreading.transduction_
    print("\nSemi-Supervised Label Spreading Results: ")
    print(f"Total samples: {len(pseudo_labels)}")
    print(f"Originally labeled: {len(y_labelled)}")
    print(f"Newly labeled: {np.sum(y_combined == -1)}")
    print(f"Label distribution: {np.bincount(pseudo_labels.astype(int))}")
    return label_spreading, pseudo_labels, X_combined

def run_neural_network(X, y):
    """
    Task 2.1: Train an MLP Neural Network classifier.
    - Split data (test_size=0.2, random_state=RANDOM_SEED)
    - Train MLPClassifier
    - Return accuracy and report
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    # Train MLP Classifier
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    
    mlp_classifier.fit(X_train, y_train)
    # Predict and evaluate
    y_pred = mlp_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("\nMLP Neural Network Results: ")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training iterations: {mlp_classifier.n_iter_}")
    print("\nClassification Report:")
    print(report)
    
    return mlp_classifier, accuracy, report, X_test, y_test, y_pred

if __name__ == "__main__":
    # Load data
    print("Running classification...")
    # Load preprocessed data
    print("Loading data...")
    
    # Check if clustered data exists
    clustered_path = "../../aura_audit/data/processed/support_logs_clustered.csv"
    if not os.path.exists(clustered_path):
        print("Please run pipeline.py first to generate clustered data.")
        exit(1)
    
    df = pd.read_csv(clustered_path)
    print(f"Loaded {len(df)} support logs")
    
    # Prepare features using TF-IDF
    print("\nPreparing TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words='english',
        ngram_range=(1, 2),
        random_state=RANDOM_SEED
    )
    
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    
    # Use cluster labels as pseudo-labels for training
    y = df['cluster'].values
    
    # Task 1.2: Run supervised baseline
    print("\n" + "="*60)
    print("TASK 1.2: SUPERVISED BASELINE (Random Forest)")
    print("="*60)
    rf_model, rf_acc, rf_report, X_test, y_test, y_pred = run_supervised_baseline(X, y)
    
    # Task 1.3: Semi-supervised learning
    print("\n" + "="*60)
    print("TASK 1.3: SEMI-SUPERVISED LEARNING (Label Spreading)")
    print("="*60)
    
    # Simulate having only 20% labeled data
    n_labeled = int(len(X) * 0.2)
    indices = np.random.RandomState(RANDOM_SEED).permutation(len(X))
    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]
    
    X_labeled = X[labeled_indices]
    y_labeled = y[labeled_indices]
    X_unlabeled = X[unlabeled_indices]
    
    ls_model, pseudo_labels, X_combined = run_semi_supervised_learning(
        X_unlabeled, X_labeled, y_labeled
    )
    
    # Task 2.1: Neural Network
    print("\n" + "="*60)
    print("TASK 2.1: NEURAL NETWORK (MLP)")
    print("="*60)
    mlp_model, mlp_acc, mlp_report, mlp_X_test, mlp_y_test, mlp_y_pred = run_neural_network(X, y)
    
    # Saving models
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(mlp_model, 'models/mlp_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("\n" + "="*60)
    print("MODELS SAVED SUCCESSFULLY")
    print("="*60)
    print("Models saved in 'models/' directory:")
    print("- random_forest_model.pkl")
    print("- mlp_model.pkl")
    print("- tfidf_vectorizer.pkl")
