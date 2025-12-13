"""
Machine Learning with Scikit-learn
===================================

Introduction to machine learning using scikit-learn.
Covers classification, regression, clustering, and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression, make_blobs

print("=" * 70)
print("MACHINE LEARNING WITH SCIKIT-LEARN")
print("=" * 70)

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================

print("\n" + "=" * 70)
print("1. DATA PREPARATION")
print("=" * 70)

# Generate synthetic classification dataset
np.random.seed(42)
X_class, y_class = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

print(f"Classification dataset shape: {X_class.shape}")
print(f"Target distribution: {np.bincount(y_class)}")

# Generate synthetic regression dataset
X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=10,
    random_state=42
)

print(f"\nRegression dataset shape: {X_reg.shape}")
print(f"Target range: [{y_reg.min():.2f}, {y_reg.max():.2f}]")

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "=" * 70)
print("2. TRAIN-TEST SPLIT")
print("=" * 70)

# Split classification data
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling applied")
print(f"Mean of scaled features: {X_train_scaled.mean(axis=0)[:3]}")
print(f"Std of scaled features: {X_train_scaled.std(axis=0)[:3]}")

# ============================================================================
# 3. CLASSIFICATION: LOGISTIC REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("3. LOGISTIC REGRESSION (CLASSIFICATION)")
print("=" * 70)

# Train model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred = log_reg.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# ============================================================================
# 4. CLASSIFICATION: DECISION TREE
# ============================================================================

print("\n" + "=" * 70)
print("4. DECISION TREE CLASSIFIER")
print("=" * 70)

# Train model
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_clf.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_clf.predict(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# Feature importance
feature_importance = dt_clf.feature_importances_
print(f"\nTop 5 important features:")
top_indices = np.argsort(feature_importance)[-5:][::-1]
for idx in top_indices:
    print(f"  Feature {idx}: {feature_importance[idx]:.4f}")

# ============================================================================
# 5. CLASSIFICATION: RANDOM FOREST
# ============================================================================

print("\n" + "=" * 70)
print("5. RANDOM FOREST CLASSIFIER")
print("=" * 70)

# Train model
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_clf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_clf.predict(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")

# Cross-validation
cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# 6. CLASSIFICATION: SUPPORT VECTOR MACHINE
# ============================================================================

print("\n" + "=" * 70)
print("6. SUPPORT VECTOR MACHINE (SVM)")
print("=" * 70)

# Train model
svm_clf = SVC(kernel='rbf', random_state=42)
svm_clf.fit(X_train_scaled, y_train)

# Predictions
y_pred_svm = svm_clf.predict(X_test_scaled)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_svm):.4f}")

# ============================================================================
# 7. CLASSIFICATION: K-NEAREST NEIGHBORS
# ============================================================================

print("\n" + "=" * 70)
print("7. K-NEAREST NEIGHBORS (KNN)")
print("=" * 70)

# Train model
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train)

# Predictions
y_pred_knn = knn_clf.predict(X_test_scaled)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_knn):.4f}")

# Find optimal K
k_range = range(1, 21)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    k_scores.append(scores.mean())

optimal_k = k_range[np.argmax(k_scores)]
print(f"\nOptimal K: {optimal_k}")
print(f"Best CV score: {max(k_scores):.4f}")

# ============================================================================
# 8. REGRESSION: LINEAR REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("8. LINEAR REGRESSION")
print("=" * 70)

# Split regression data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train model
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = lin_reg.predict(X_test_reg)

# Evaluation
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

print(f"\nModel coefficients (first 5): {lin_reg.coef_[:5]}")
print(f"Intercept: {lin_reg.intercept_:.2f}")

# ============================================================================
# 9. CLUSTERING: K-MEANS
# ============================================================================

print("\n" + "=" * 70)
print("9. K-MEANS CLUSTERING")
print("=" * 70)

# Generate clustering data
X_cluster, y_cluster = make_blobs(
    n_samples=300, 
    n_features=2, 
    centers=3, 
    cluster_std=0.5,
    random_state=42
)

# Train K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster)

print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"\nInertia (sum of squared distances): {kmeans.inertia_:.2f}")

# Count samples per cluster
unique, counts = np.unique(clusters, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"Cluster {cluster_id}: {count} samples")

# Elbow method to find optimal K
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(X_cluster)
    inertias.append(kmeans_temp.inertia_)

print(f"\nInertia values for K=1 to 10:")
for k, inertia in zip(K_range, inertias):
    print(f"  K={k}: {inertia:.2f}")

# ============================================================================
# 10. MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("10. MODEL COMPARISON")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = []

for name, model in models.items():
    # Train
    if name in ['Logistic Regression', 'SVM', 'KNN']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'F1-Score': f1
    })

# Display results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)
print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))

# ============================================================================
# 11. SAVING AND LOADING MODELS
# ============================================================================

print("\n" + "=" * 70)
print("11. SAVING AND LOADING MODELS")
print("=" * 70)

import pickle

# Save model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)
print("Model saved to 'random_forest_model.pkl'")

# Load model
with open('random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
print("Model loaded successfully")

# Verify loaded model
y_pred_loaded = loaded_model.predict(X_test)
print(f"Loaded model accuracy: {accuracy_score(y_test, y_pred_loaded):.4f}")

# ============================================================================
# 12. PRACTICAL EXAMPLE: COMPLETE ML PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("12. COMPLETE ML PIPELINE EXAMPLE")
print("=" * 70)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Predict
y_pred_pipeline = pipeline.predict(X_test)

# Evaluate
print(f"Pipeline Accuracy: {accuracy_score(y_test, y_pred_pipeline):.4f}")
print(f"Pipeline F1-Score: {f1_score(y_test, y_pred_pipeline):.4f}")

print("\n" + "=" * 70)
print("SCIKIT-LEARN FUNDAMENTALS COMPLETE!")
print("=" * 70)
