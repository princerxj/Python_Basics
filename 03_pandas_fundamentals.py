"""
Pandas Fundamentals for AI/ML
==============================

Pandas is essential for data manipulation and analysis in machine learning.
It provides DataFrame and Series objects for handling structured data.
"""

import numpy as np
import pandas as pd

print("=" * 60)
print("PANDAS FUNDAMENTALS FOR AI/ML")
print("=" * 60)

# ============================================================================
# 1. CREATING SERIES AND DATAFRAMES
# ============================================================================

print("\n" + "=" * 60)
print("1. CREATING SERIES AND DATAFRAMES")
print("=" * 60)

# Series - 1D labeled array
s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print("Series:")
print(s)
print(f"\nAccess element 'c': {s['c']}")

# DataFrame - 2D labeled data structure
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Salary': [50000, 60000, 75000, 55000, 65000],
    'Department': ['HR', 'IT', 'IT', 'Finance', 'HR']
}
df = pd.DataFrame(data)
print("\nDataFrame:")
print(df)

# Create from numpy array
np.random.seed(42)
arr = np.random.randint(0, 100, size=(5, 3))
df_numpy = pd.DataFrame(arr, columns=['Feature1', 'Feature2', 'Feature3'])
print("\nDataFrame from NumPy:")
print(df_numpy)

# ============================================================================
# 2. DATAFRAME ATTRIBUTES AND INFO
# ============================================================================

print("\n" + "=" * 60)
print("2. DATAFRAME ATTRIBUTES")
print("=" * 60)

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Index: {df.index.tolist()}")
print(f"Data types:\n{df.dtypes}")
print(f"\nInfo:")
print(df.info())

print("\nFirst 3 rows:")
print(df.head(3))

print("\nLast 2 rows:")
print(df.tail(2))

print("\nStatistical summary:")
print(df.describe())

# ============================================================================
# 3. SELECTING DATA
# ============================================================================

print("\n" + "=" * 60)
print("3. SELECTING DATA")
print("=" * 60)

# Select single column
print("Select 'Name' column:")
print(df['Name'])

# Select multiple columns
print("\nSelect multiple columns:")
print(df[['Name', 'Age']])

# Select rows by index
print("\nFirst 3 rows:")
print(df[0:3])

# loc - label-based indexing
print("\nUsing loc[0]:")
print(df.loc[0])

print("\nUsing loc[0:2, ['Name', 'Age']]:")
print(df.loc[0:2, ['Name', 'Age']])

# iloc - integer-based indexing
print("\nUsing iloc[0:3, 0:2]:")
print(df.iloc[0:3, 0:2])

# Boolean indexing (filtering)
print("\nFilter: Age > 28")
print(df[df['Age'] > 28])

print("\nFilter: Department == 'IT'")
print(df[df['Department'] == 'IT'])

print("\nFilter: Age > 28 AND Department == 'IT'")
print(df[(df['Age'] > 28) & (df['Department'] == 'IT')])

# ============================================================================
# 4. ADDING AND MODIFYING DATA
# ============================================================================

print("\n" + "=" * 60)
print("4. ADDING AND MODIFYING DATA")
print("=" * 60)

# Add new column
df_copy = df.copy()
df_copy['Bonus'] = df_copy['Salary'] * 0.1
print("Added 'Bonus' column:")
print(df_copy)

# Modify existing column
df_copy['Age'] = df_copy['Age'] + 1
print("\nIncremented 'Age' by 1:")
print(df_copy[['Name', 'Age']])

# Add new row
new_row = {'Name': 'Frank', 'Age': 29, 'Salary': 58000, 'Department': 'IT', 'Bonus': 5800}
df_copy = pd.concat([df_copy, pd.DataFrame([new_row])], ignore_index=True)
print("\nAdded new row:")
print(df_copy)

# ============================================================================
# 5. HANDLING MISSING DATA
# ============================================================================

print("\n" + "=" * 60)
print("5. HANDLING MISSING DATA")
print("=" * 60)

# Create DataFrame with missing values
data_missing = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, np.nan, 8, 9],
    'C': [10, 11, 12, 13, 14]
}
df_missing = pd.DataFrame(data_missing)
print("DataFrame with missing values:")
print(df_missing)

# Check for missing values
print("\nCheck for missing values:")
print(df_missing.isnull())

print("\nCount missing values per column:")
print(df_missing.isnull().sum())

# Drop rows with missing values
print("\nDrop rows with any missing values:")
print(df_missing.dropna())

# Fill missing values
print("\nFill missing values with 0:")
print(df_missing.fillna(0))

print("\nFill missing values with mean:")
print(df_missing.fillna(df_missing.mean()))

# Forward fill
print("\nForward fill:")
print(df_missing.fillna(method='ffill'))

# ============================================================================
# 6. GROUPBY AND AGGREGATION
# ============================================================================

print("\n" + "=" * 60)
print("6. GROUPBY AND AGGREGATION")
print("=" * 60)

# Group by department
print("Group by Department and calculate mean:")
print(df.groupby('Department')[['Age', 'Salary']].mean())

print("\nGroup by Department and calculate multiple aggregations:")
print(df.groupby('Department').agg({
    'Age': 'mean',
    'Salary': ['mean', 'min', 'max']
}))

print("\nCount by Department:")
print(df['Department'].value_counts())

# ============================================================================
# 7. SORTING
# ============================================================================

print("\n" + "=" * 60)
print("7. SORTING")
print("=" * 60)

print("Sort by Age (ascending):")
print(df.sort_values('Age'))

print("\nSort by Salary (descending):")
print(df.sort_values('Salary', ascending=False))

print("\nSort by multiple columns:")
print(df.sort_values(['Department', 'Age']))

# ============================================================================
# 8. MERGING AND JOINING
# ============================================================================

print("\n" + "=" * 60)
print("8. MERGING AND JOINING")
print("=" * 60)

# Create sample DataFrames
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David']
})

df2 = pd.DataFrame({
    'ID': [1, 2, 3, 5],
    'Score': [85, 90, 78, 88]
})

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Inner join
print("\nInner join:")
print(pd.merge(df1, df2, on='ID', how='inner'))

# Left join
print("\nLeft join:")
print(pd.merge(df1, df2, on='ID', how='left'))

# Outer join
print("\nOuter join:")
print(pd.merge(df1, df2, on='ID', how='outer'))

# ============================================================================
# 9. APPLY FUNCTIONS
# ============================================================================

print("\n" + "=" * 60)
print("9. APPLY FUNCTIONS")
print("=" * 60)

# Apply function to column
df_apply = df.copy()
df_apply['Salary_Normalized'] = df_apply['Salary'].apply(lambda x: x / 100000)
print("Applied normalization to Salary:")
print(df_apply[['Name', 'Salary', 'Salary_Normalized']])

# Apply function to multiple columns
def categorize_age(age):
    if age < 30:
        return 'Young'
    elif age < 35:
        return 'Middle'
    else:
        return 'Senior'

df_apply['Age_Category'] = df_apply['Age'].apply(categorize_age)
print("\nCategorized Age:")
print(df_apply[['Name', 'Age', 'Age_Category']])

# ============================================================================
# 10. READING AND WRITING DATA
# ============================================================================

print("\n" + "=" * 60)
print("10. READING AND WRITING DATA")
print("=" * 60)

# Save to CSV
df.to_csv('employee_data.csv', index=False)
print("DataFrame saved to 'employee_data.csv'")

# Read from CSV
df_loaded = pd.read_csv('employee_data.csv')
print("\nLoaded DataFrame from CSV:")
print(df_loaded)

# Save to Excel (requires openpyxl)
try:
    df.to_excel('employee_data.xlsx', index=False)
    print("\nDataFrame saved to 'employee_data.xlsx'")
except ImportError:
    print("\nNote: Install 'openpyxl' to save Excel files: pip install openpyxl")

# ============================================================================
# 11. PRACTICAL ML EXAMPLE: Feature Engineering
# ============================================================================

print("\n" + "=" * 60)
print("11. PRACTICAL ML EXAMPLE: FEATURE ENGINEERING")
print("=" * 60)

# Create sample dataset
np.random.seed(42)
ml_data = pd.DataFrame({
    'Transaction_Amount': np.random.uniform(10, 1000, 100),
    'Transaction_Count': np.random.randint(1, 50, 100),
    'Customer_Age': np.random.randint(18, 80, 100),
    'Account_Balance': np.random.uniform(100, 10000, 100)
})

print("Original dataset (first 5 rows):")
print(ml_data.head())

# Create new features
ml_data['Avg_Transaction'] = ml_data['Transaction_Amount'] / ml_data['Transaction_Count']
ml_data['Balance_to_Transaction_Ratio'] = ml_data['Account_Balance'] / ml_data['Transaction_Amount']
ml_data['Age_Group'] = pd.cut(ml_data['Customer_Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])

print("\nDataset with engineered features (first 5 rows):")
print(ml_data.head())

print("\nStatistical summary of engineered features:")
print(ml_data[['Avg_Transaction', 'Balance_to_Transaction_Ratio']].describe())

print("\nDistribution by Age Group:")
print(ml_data['Age_Group'].value_counts())

# ============================================================================
# 12. PRACTICAL ML EXAMPLE: Data Preparation
# ============================================================================

print("\n" + "=" * 60)
print("12. PRACTICAL ML EXAMPLE: DATA PREPARATION")
print("=" * 60)

# One-hot encoding for categorical variables
ml_data_encoded = pd.get_dummies(ml_data, columns=['Age_Group'], prefix='Age')
print("One-hot encoded data (first 5 rows):")
print(ml_data_encoded.head())

# Normalize numerical features
from sklearn.preprocessing import MinMaxScaler

numerical_cols = ['Transaction_Amount', 'Transaction_Count', 'Customer_Age', 'Account_Balance']
scaler = MinMaxScaler()
ml_data_normalized = ml_data.copy()
ml_data_normalized[numerical_cols] = scaler.fit_transform(ml_data[numerical_cols])

print("\nNormalized data (first 5 rows):")
print(ml_data_normalized.head())

print("\n" + "=" * 60)
print("PANDAS FUNDAMENTALS COMPLETE!")
print("=" * 60)
