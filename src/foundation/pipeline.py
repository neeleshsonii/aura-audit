import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def clean_text(text):
    """
    Task 1.1: Implement text normalization and PII removal.
    - Convert to lowercase
    - Remove email addresses and phone numbers
    - Strip extra whitespace
    """
    # TODO: Candidate implements this
    pass

def discover_intents(df, n_clusters=5):
    """
    Task 1.1: Use K-Means to identify clusters of support issues.
    - Convert text to numeric (hint: use TF-IDF or simple embeddings)
    - Apply KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    - Return the cluster labels
    """
    # TODO: Candidate implements this
    pass

def plot_clusters(df, labels):
    """
    Task 1.1: Use PCA to visualize clusters in 2D.
    """
    # TODO: Candidate implements this
    pass

if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv("../../data/raw/support_logs.csv")
    print("Preprocessing data...")
    # df['clean_text'] = df['text'].apply(clean_text)
    # labels = discover_intents(df)
    # plot_clusters(df, labels)
