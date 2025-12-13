"""
Data Visualization with Matplotlib for AI/ML
=============================================

Matplotlib is the foundational plotting library in Python.
Essential for visualizing data, model performance, and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("=" * 60)
print("MATPLOTLIB FOR AI/ML")
print("=" * 60)

# ============================================================================
# 1. LINE PLOTS
# ============================================================================

print("\n1. Creating Line Plots...")

# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 4))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True)
plt.savefig('01_line_plot.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 01_line_plot.png")

# Multiple lines
plt.figure(figsize=(10, 4))
plt.plot(x, np.sin(x), label='sin(x)', linewidth=2)
plt.plot(x, np.cos(x), label='cos(x)', linewidth=2, linestyle='--')
plt.title('Trigonometric Functions')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('02_multiple_lines.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 02_multiple_lines.png")

# ============================================================================
# 2. SCATTER PLOTS
# ============================================================================

print("\n2. Creating Scatter Plots...")

# Simple scatter plot
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6)
plt.title('Scatter Plot Example')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.savefig('03_scatter_plot.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 03_scatter_plot.png")

# Colored scatter plot (classification visualization)
np.random.seed(42)
X1 = np.random.randn(50, 2) + np.array([2, 2])
X2 = np.random.randn(50, 2) + np.array([-2, -2])

plt.figure(figsize=(8, 6))
plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='Class 1', alpha=0.6)
plt.scatter(X2[:, 0], X2[:, 1], c='red', label='Class 2', alpha=0.6)
plt.title('Binary Classification Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('04_classification_scatter.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 04_classification_scatter.png")

# ============================================================================
# 3. BAR PLOTS
# ============================================================================

print("\n3. Creating Bar Plots...")

# Simple bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='steelblue', alpha=0.7)
plt.title('Bar Plot Example')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('05_bar_plot.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 05_bar_plot.png")

# Grouped bar plot (comparing models)
models = ['Model A', 'Model B', 'Model C']
accuracy = [0.85, 0.92, 0.88]
precision = [0.83, 0.90, 0.86]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, accuracy, width, label='Accuracy', alpha=0.8)
plt.bar(x + width/2, precision, width, label='Precision', alpha=0.8)
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, models)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('06_grouped_bar.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 06_grouped_bar.png")

# ============================================================================
# 4. HISTOGRAMS
# ============================================================================

print("\n4. Creating Histograms...")

# Simple histogram
np.random.seed(42)
data = np.random.randn(1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('07_histogram.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 07_histogram.png")

# Multiple histograms (comparing distributions)
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1.5, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data1, bins=30, alpha=0.5, label='Distribution 1', color='blue')
plt.hist(data2, bins=30, alpha=0.5, label='Distribution 2', color='red')
plt.title('Comparing Two Distributions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('08_multiple_histograms.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 08_multiple_histograms.png")

# ============================================================================
# 5. BOX PLOTS
# ============================================================================

print("\n5. Creating Box Plots...")

# Box plot (useful for showing distributions and outliers)
np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3'])
plt.title('Box Plot Comparison')
plt.ylabel('Values')
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('09_box_plot.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 09_box_plot.png")

# ============================================================================
# 6. HEATMAPS (Correlation Matrix)
# ============================================================================

print("\n6. Creating Heatmaps...")

# Correlation matrix heatmap
np.random.seed(42)
data = pd.DataFrame({
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100),
    'Feature3': np.random.randn(100),
    'Feature4': np.random.randn(100)
})
data['Feature2'] = data['Feature1'] + np.random.randn(100) * 0.5
correlation = data.corr()

plt.figure(figsize=(8, 6))
plt.imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title('Correlation Matrix Heatmap')

# Add correlation values as text
for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(j, i, f'{correlation.iloc[i, j]:.2f}', 
                ha='center', va='center', color='white' if abs(correlation.iloc[i, j]) > 0.5 else 'black')

plt.tight_layout()
plt.savefig('10_heatmap.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 10_heatmap.png")

# ============================================================================
# 7. SUBPLOTS
# ============================================================================

print("\n7. Creating Subplots...")

# Multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sin Wave')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scatter plot
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50), alpha=0.6)
axes[0, 1].set_title('Scatter Plot')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Bar plot
axes[1, 0].bar(['A', 'B', 'C'], [3, 7, 5], color='steelblue', alpha=0.7)
axes[1, 0].set_title('Bar Plot')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Histogram
axes[1, 1].hist(np.random.randn(1000), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Histogram')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('11_subplots.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 11_subplots.png")

# ============================================================================
# 8. ML EXAMPLE: Training History Visualization
# ============================================================================

print("\n8. ML Example: Training History...")

# Simulate training history
epochs = np.arange(1, 51)
train_loss = 2 * np.exp(-epochs/10) + 0.1 * np.random.randn(50)
val_loss = 2 * np.exp(-epochs/10) + 0.2 * np.random.randn(50)
train_acc = 1 - np.exp(-epochs/10) + 0.05 * np.random.randn(50)
val_acc = 1 - np.exp(-epochs/10) + 0.08 * np.random.randn(50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1.plot(epochs, train_loss, label='Training Loss', linewidth=2)
ax1.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Model Loss During Training')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(epochs, train_acc, label='Training Accuracy', linewidth=2)
ax2.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Model Accuracy During Training')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('12_training_history.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 12_training_history.png")

# ============================================================================
# 9. ML EXAMPLE: Confusion Matrix
# ============================================================================

print("\n9. ML Example: Confusion Matrix...")

# Simulate confusion matrix
confusion_matrix = np.array([
    [85, 10, 5],
    [8, 88, 4],
    [7, 5, 88]
])

classes = ['Class 0', 'Class 1', 'Class 2']

plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, cmap='Blues', aspect='auto')
plt.colorbar(label='Count')
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Add values as text
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(confusion_matrix[i, j]), 
                ha='center', va='center', 
                color='white' if confusion_matrix[i, j] > 50 else 'black',
                fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('13_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 13_confusion_matrix.png")

# ============================================================================
# 10. ML EXAMPLE: Feature Importance
# ============================================================================

print("\n10. ML Example: Feature Importance...")

# Simulate feature importance
features = ['Age', 'Income', 'Credit Score', 'Years Employed', 'Debt Ratio', 
           'Loan Amount', 'Property Value', 'Education']
importance = np.array([0.15, 0.22, 0.18, 0.12, 0.10, 0.08, 0.09, 0.06])

# Sort by importance
indices = np.argsort(importance)
features_sorted = [features[i] for i in indices]
importance_sorted = importance[indices]

plt.figure(figsize=(10, 6))
plt.barh(features_sorted, importance_sorted, color='steelblue', alpha=0.8)
plt.xlabel('Importance Score')
plt.title('Feature Importance in ML Model')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('14_feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 14_feature_importance.png")

# ============================================================================
# 11. ML EXAMPLE: Decision Boundary
# ============================================================================

print("\n11. ML Example: Decision Boundary...")

# Create a mesh grid
np.random.seed(42)
h = 0.1
x_min, x_max = -3, 3
y_min, y_max = -3, 3
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Simulate decision boundary (simple circular boundary)
Z = np.sqrt(xx**2 + yy**2) < 2

# Generate sample points
X1 = np.random.randn(100, 2) * 0.8
X2 = np.random.randn(100, 2) * 0.8 + np.array([2.5, 0])

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X2[:, 0], X2[:, 1], c='red', label='Class 1', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary Visualization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('15_decision_boundary.png', dpi=100, bbox_inches='tight')
plt.close()
print("   Saved: 15_decision_boundary.png")

print("\n" + "=" * 60)
print("MATPLOTLIB VISUALIZATION COMPLETE!")
print(f"Generated 15 visualization examples")
print("=" * 60)
