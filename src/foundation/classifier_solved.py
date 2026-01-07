import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def run_supervised_baseline(df):
    """
    Task 1.4: Train a Supervised Random Forest.
    """
    # Using the 'actual_intent' for the gold standard evaluation in this solution
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['actual_intent']
    
    clf = RandomForestClassifier(random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Supervised Classification Report:")
    print(classification_report(y_test, y_pred))
    return clf, vectorizer

def run_semi_supervised_learning(df):
    """
    Task 1.3: Use LabelSpreading to propagate labels.
    """
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['text']).toarray()
    
    # Simulate a scenario where only 10% of labels are known
    labels = np.copy(df['actual_intent'].values)
    rng = np.random.RandomState(RANDOM_SEED)
    random_unlabeled_points = rng.rand(len(labels)) < 0.9
    
    # Factorize labels for LabelSpreading (it requires integers)
    y_encoded, y_categories = pd.factorize(labels)
    y_lp = np.copy(y_encoded)
    y_lp[random_unlabeled_points] = -1
    
    print(f"Running LabelSpreading with {sum(y_lp != -1)} initial labels...")
    lp_model.fit(X, y_lp)
    
    predicted_labels = lp_model.transduction_
    accuracy = accuracy_score(y_encoded, predicted_labels)
    print(f"Semi-supervised Accuracy: {accuracy:.4f}")
    
    return predicted_labels, y_categories

if __name__ == "__main__":
    # Load the reference labeled data for solution verification
    
    print("Step 3: Running Semi-supervised Label Spreading...")
    run_semi_supervised_learning(df)
    
    print("\nStep 4: Running Supervised Random Forest Baseline...")
    run_supervised_baseline(df)
