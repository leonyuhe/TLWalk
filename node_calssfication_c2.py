import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from torch_geometric.datasets import PolBlogs
from torch_geometric.utils import to_networkx
from karateclub import (
    DeepWalk, Node2Vec, Role2Vec, Comm2Vec,
    Walklets, TwoLayerCommunityWalk, MNMF,CommunityRoleWalk
)

# Load PolBlogs dataset
dataset = PolBlogs(root="./data/PolBlogs")
data = dataset[0]

# Convert PyTorch Geometric graph to NetworkX graph
graph = to_networkx(data, to_undirected=True)

# Extract labels (0 for liberal, 1 for conservative)
y = data.y.numpy()

# Define embedding methods
methods = {
    # "DeepWalk": DeepWalk(),
    # "Node2Vec": Node2Vec(),
    "CommunityRoleWalk":CommunityRoleWalk(),
    "TwoLayerCommunityWalk": TwoLayerCommunityWalk(),
    "MNMF": MNMF(),
}

# Evaluate each method
results = {}
for method_name, model in methods.items():
    print(f"Testing {method_name}...")

    # Train the embedding model
    model.fit(graph)
    X = model.get_embedding()

    # Ensure `X` has the correct shape
    if X.shape[0] != len(y):
        raise ValueError(f"Mismatch: {X.shape[0]} nodes in X, but {len(y)} labels in y")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train a logistic regression classifier with binary classification
    downstream_model = LogisticRegression(
        random_state=0, max_iter=2000
    )
    downstream_model.fit(X_train, y_train)

    # Predict and calculate AUC
    y_hat = downstream_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_hat)

    results[method_name] = auc
    print(f"{method_name} AUC: {auc:.4f}")

# Print all results
print("\nSummary of AUC scores:")
for method, auc in results.items():
    print(f"{method}: {auc:.4f}")