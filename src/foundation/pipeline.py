import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import os

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def clean_text(text):
    """
    Task 1.1: Implement text normalization and PII removal.
    - Convert to lowercase
    - Remove email addresses and phone numbers
    - Strip extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove email addresses (PII)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Remove phone numbers (PII) - various formats
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{3}-\d{4}\b', '[PHONE]', text)
    # Remove credit card-like numbers (PII)
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', text)
    # Remove pin codes or small numbers that might be PII
    text = re.sub(r'pin code:?\s*\d+', '[PIN]', text, flags=re.IGNORECASE)
    text = re.sub(r'password:?\s*\S+', '[PASSWORD]', text, flags=re.IGNORECASE)
    # Remove invoice/order numbers for consistency
    text = re.sub(r'\binvoice\s+\d+\b', 'invoice [NUMBER]', text)
    text = re.sub(r'\border\s+\d+\b', 'order [NUMBER]', text)
    # Strip extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[!]{2,}', '!', text)  # Reduce multiple exclamations
    text = text.strip()
    return text

def discover_intents(df, n_clusters=5):
    """
    Task 1.1: Use K-Means to identify clusters of support issues.
    - Convert text to numeric (hint: use TF-IDF or simple embeddings)
    - Apply KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    - Return the cluster labels
    """
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),
        random_state=RANDOM_SEED
    )
    X = vectorizer.fit_transform(df['clean_text'])
    
    # Apply K-Means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        n_init=10
    )
    labels = kmeans.fit_predict(X)
    # Store the vectorizer and features for later use
    df['cluster'] = labels
    return labels, vectorizer, X

def plot_clusters(df, labels):
    """
    Task 1.1: Use PCA to visualize clusters in 2D.
    """
    if X is None:
        return
    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca.fit_transform(X.toarray())
    # Create plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('K-Means Clustering of Support Logs (PCA Visualization)')
    
    # Save plot
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/cluster_visualization.png', dpi=150, bbox_inches='tight')
    print("Cluster visualization saved to outputs/cluster_visualization.png")
    plt.close()

if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv("../../aura_audit/data/raw/support_logs.csv")
    print(f"Loaded {len(df)} support logs")
    
    print("Step 1: Cleaning text and removing PII...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    print("Step 2: Discovering intents using K-Means clustering...")
    labels, vectorizer, X = discover_intents(df, n_clusters=5)
    
    print("Step 3: Visualizing clusters...")
    plot_clusters(df, labels, X)
    # Save processed data
    os.makedirs('../../aura_audit/data/processed', exist_ok=True)
    df.to_csv('../../aura_audit/data/processed/support_logs_clustered.csv', index=False)
    print(f"Processed data saved with {len(df['cluster'].unique())} clusters identified")
    # Print cluster statistics
    print("\nCluster Distribution:")
    print(df['cluster'].value_counts().sort_index())
