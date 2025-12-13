# Quick Reference Guide for Python AI/ML

## Installation

```bash
# Install required packages
pip install -r requirements.txt

# For deep learning (optional)
pip install tensorflow
```

## File Overview

| File | Topic | Key Concepts |
|------|-------|--------------|
| `01_python_basics.py` | Python Fundamentals | Variables, data structures, functions, OOP |
| `02_numpy_fundamentals.py` | NumPy | Arrays, operations, linear algebra |
| `03_pandas_fundamentals.py` | Pandas | DataFrames, data manipulation, preprocessing |
| `04_matplotlib_visualization.py` | Visualization | Plots, charts, ML visualizations |
| `05_scikit_learn_ml.py` | Machine Learning | Classification, regression, clustering |
| `06_deep_learning_intro.py` | Deep Learning | Neural networks, CNNs, TensorFlow |

## Quick Start Examples

### Data Manipulation
```python
import pandas as pd
import numpy as np

# Create DataFrame
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    'label': [0, 1, 0, 1, 0]
})

# Basic operations
print(df.head())
print(df.describe())
print(df[df['label'] == 1])
```

### Simple Machine Learning
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

### Data Visualization
```python
import matplotlib.pyplot as plt

# Line plot
plt.plot(x, y)
plt.title('My Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Scatter plot
plt.scatter(x, y, alpha=0.5)
plt.show()

# Histogram
plt.hist(data, bins=30)
plt.show()
```

## Common Workflows

### 1. Data Analysis Workflow
```
Load Data → Explore → Clean → Transform → Analyze → Visualize
```

### 2. Machine Learning Workflow
```
Collect Data → Preprocess → Split → Train → Evaluate → Deploy
```

### 3. Feature Engineering
```python
# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'])

# Create new features
df['new_feature'] = df['feature1'] * df['feature2']
```

## Essential Commands

### NumPy
```python
# Create array
arr = np.array([1, 2, 3, 4, 5])

# Common operations
arr.mean()  # Average
arr.std()   # Standard deviation
arr.max()   # Maximum
arr.min()   # Minimum
arr.reshape(5, 1)  # Reshape

# Linear algebra
np.dot(A, B)  # Matrix multiplication
np.linalg.inv(A)  # Matrix inverse
```

### Pandas
```python
# Read/write data
df = pd.read_csv('data.csv')
df.to_csv('output.csv', index=False)

# Select data
df['column']  # Single column
df[['col1', 'col2']]  # Multiple columns
df[df['col'] > 5]  # Filter rows

# Grouping
df.groupby('category').mean()
df.groupby('category').agg({'col1': 'mean', 'col2': 'sum'})

# Handle missing data
df.dropna()  # Drop missing values
df.fillna(0)  # Fill with value
```

### Scikit-learn
```python
# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Clustering
from sklearn.cluster import KMeans

# Model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
```

## Common Pitfalls to Avoid

1. **Not splitting data properly** - Always split before preprocessing
2. **Data leakage** - Don't fit scaler on test data
3. **Overfitting** - Use regularization and validation
4. **Imbalanced datasets** - Use appropriate metrics and techniques
5. **Not normalizing features** - Scale features for better performance
6. **Ignoring missing values** - Handle them explicitly

## Performance Tips

1. **Use vectorized operations** instead of loops
2. **Use appropriate data types** (int32 vs int64)
3. **Batch processing** for large datasets
4. **Use generators** for memory efficiency
5. **Leverage GPU** for deep learning (if available)

## Next Steps

1. Practice with real datasets (Kaggle, UCI ML Repository)
2. Build end-to-end projects
3. Participate in competitions
4. Read research papers
5. Contribute to open-source projects

## Useful Resources

- **Kaggle**: https://www.kaggle.com/
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/
- **Papers with Code**: https://paperswithcode.com/
- **Towards Data Science**: https://towardsdatascience.com/
- **Machine Learning Mastery**: https://machinelearningmastery.com/

## Debugging Tips

```python
# Check data shapes
print(X.shape, y.shape)

# Check data types
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())

# Statistical summary
print(df.describe())

# Check unique values
print(df['column'].unique())
print(df['column'].value_counts())
```

---

Remember: Practice is key! Run the examples, modify them, and experiment with your own data.
