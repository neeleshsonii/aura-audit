import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import re

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def clean_text(text):
    """
    Task 1.1: Implement text normalization and PII removal.
    """
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    # Strip extra whitespace
    text = re.strip(text)
    return text

def discover_intents(df, n_clusters=5):
    """
    Task 1.1: Use K-Means to identify clusters of support issues.
    """
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    
    return labels, X

def plot_clusters(X, labels):
    """
    Task 1.1: Use PCA to visualize clusters in 2D.
    """
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    plt.figure(figsize=(10, 7))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
    plt.title("Cluster Discovery (PCA visualization)")
    plt.colorbar(label='Cluster ID')
    plt.savefig("cluster_visualization.png")
    print("Cluster plot saved as cluster_visualization.png")

if __name__ == "__main__":
    # Load raw data
    data_path = "../../data/raw/support_logs.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Run generate_data.py first.")
        exit()

    print("Step 1: Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    print("Step 2: Discovering intents via Unsupervised Clustering...")
    labels, X_tfidf = discover_intents(df)
    df['cluster_label'] = labels
    
    print("Step 2.1: Visualizing clusters...")
    plot_clusters(X_tfidf, labels)
    
    # Save processed data for the next step
    df.to_csv("../../data/raw/support_logs_with_clusters.csv", index=False)
    print("Processed logs saved.")
