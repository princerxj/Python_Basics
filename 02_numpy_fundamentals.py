"""
NumPy Fundamentals for AI/ML
=============================

NumPy is the fundamental package for numerical computing in Python.
Essential for handling arrays and matrices in machine learning.
"""

import numpy as np

print("=" * 60)
print("NUMPY FUNDAMENTALS FOR AI/ML")
print("=" * 60)

# ============================================================================
# 1. CREATING ARRAYS
# ============================================================================

print("\n" + "=" * 60)
print("1. CREATING ARRAYS")
print("=" * 60)

# From Python lists
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"1D Array: {arr1d}")
print(f"2D Array:\n{arr2d}")

# Special arrays
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
identity = np.eye(3)
print(f"\nZeros array:\n{zeros}")
print(f"\nOnes array:\n{ones}")
print(f"\nIdentity matrix:\n{identity}")

# Range arrays
range_arr = np.arange(0, 10, 2)  # start, stop, step
linspace_arr = np.linspace(0, 1, 5)  # start, stop, num_points
print(f"\nRange array: {range_arr}")
print(f"Linspace array: {linspace_arr}")

# Random arrays (very important in ML!)
np.random.seed(42)  # for reproducibility
random_arr = np.random.rand(3, 3)  # uniform [0, 1)
normal_arr = np.random.randn(3, 3)  # standard normal
random_int = np.random.randint(0, 10, size=(3, 3))
print(f"\nRandom uniform:\n{random_arr}")
print(f"\nRandom normal:\n{normal_arr}")
print(f"\nRandom integers:\n{random_int}")

# ============================================================================
# 2. ARRAY ATTRIBUTES
# ============================================================================

print("\n" + "=" * 60)
print("2. ARRAY ATTRIBUTES")
print("=" * 60)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Array:\n{arr}")
print(f"Shape: {arr.shape}")  # (rows, columns)
print(f"Dimensions: {arr.ndim}")
print(f"Size (total elements): {arr.size}")
print(f"Data type: {arr.dtype}")
print(f"Item size (bytes): {arr.itemsize}")

# ============================================================================
# 3. ARRAY INDEXING AND SLICING
# ============================================================================

print("\n" + "=" * 60)
print("3. INDEXING AND SLICING")
print("=" * 60)

# 1D indexing
arr1d = np.array([10, 20, 30, 40, 50])
print(f"Array: {arr1d}")
print(f"First element: {arr1d[0]}")
print(f"Last element: {arr1d[-1]}")
print(f"Slice [1:4]: {arr1d[1:4]}")

# 2D indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\nArray:\n{arr2d}")
print(f"Element at [0, 0]: {arr2d[0, 0]}")
print(f"Element at [1, 2]: {arr2d[1, 2]}")
print(f"First row: {arr2d[0, :]}")
print(f"Second column: {arr2d[:, 1]}")
print(f"Subarray:\n{arr2d[0:2, 1:3]}")

# Boolean indexing (very useful for filtering data!)
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = data > 5
print(f"\nData: {data}")
print(f"Mask (data > 5): {mask}")
print(f"Filtered data: {data[mask]}")

# ============================================================================
# 4. ARRAY OPERATIONS
# ============================================================================

print("\n" + "=" * 60)
print("4. ARRAY OPERATIONS")
print("=" * 60)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"Array a: {a}")
print(f"Array b: {b}")

# Element-wise operations
print(f"\nAddition: {a + b}")
print(f"Subtraction: {a - b}")
print(f"Multiplication: {a * b}")
print(f"Division: {a / b}")
print(f"Power: {a ** 2}")

# Scalar operations
print(f"\na + 10: {a + 10}")
print(f"a * 2: {a * 2}")

# Universal functions
arr = np.array([1, 4, 9, 16, 25])
print(f"\nArray: {arr}")
print(f"Square root: {np.sqrt(arr)}")
print(f"Exponential: {np.exp(a)}")
print(f"Logarithm: {np.log(arr)}")
print(f"Sine: {np.sin(a)}")

# ============================================================================
# 5. AGGREGATION FUNCTIONS
# ============================================================================

print("\n" + "=" * 60)
print("5. AGGREGATION FUNCTIONS")
print("=" * 60)

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Data:\n{data}")

print(f"\nSum: {np.sum(data)}")
print(f"Sum along axis 0 (columns): {np.sum(data, axis=0)}")
print(f"Sum along axis 1 (rows): {np.sum(data, axis=1)}")

print(f"\nMean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Standard deviation: {np.std(data)}")
print(f"Variance: {np.var(data)}")

print(f"\nMin: {np.min(data)}")
print(f"Max: {np.max(data)}")
print(f"Argmin (index): {np.argmin(data)}")
print(f"Argmax (index): {np.argmax(data)}")

# ============================================================================
# 6. ARRAY MANIPULATION
# ============================================================================

print("\n" + "=" * 60)
print("6. ARRAY MANIPULATION")
print("=" * 60)

arr = np.arange(12)
print(f"Original array: {arr}")

# Reshaping
reshaped = arr.reshape(3, 4)
print(f"\nReshaped (3x4):\n{reshaped}")

reshaped2 = arr.reshape(4, 3)
print(f"\nReshaped (4x3):\n{reshaped2}")

# Flattening
flattened = reshaped.flatten()
print(f"\nFlattened: {flattened}")

# Transpose
print(f"\nTranspose:\n{reshaped.T}")

# Concatenation
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(f"\nArray a:\n{a}")
print(f"Array b:\n{b}")

vstack = np.vstack((a, b))  # vertical stack
hstack = np.hstack((a, b))  # horizontal stack
print(f"\nVertical stack:\n{vstack}")
print(f"Horizontal stack:\n{hstack}")

# ============================================================================
# 7. LINEAR ALGEBRA (Critical for ML!)
# ============================================================================

print("\n" + "=" * 60)
print("7. LINEAR ALGEBRA")
print("=" * 60)

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")

# Dot product
dot_product = np.dot(A, B)
print(f"\nDot product (A @ B):\n{dot_product}")

# Matrix @ operator (Python 3.5+)
matmul = A @ B
print(f"Matrix multiplication (A @ B):\n{matmul}")

# Vector dot product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"\nVector v1: {v1}")
print(f"Vector v2: {v2}")
print(f"Dot product: {np.dot(v1, v2)}")

# Matrix determinant
det = np.linalg.det(A)
print(f"\nDeterminant of A: {det}")

# Matrix inverse
inv = np.linalg.inv(A)
print(f"Inverse of A:\n{inv}")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# ============================================================================
# 8. BROADCASTING
# ============================================================================

print("\n" + "=" * 60)
print("8. BROADCASTING")
print("=" * 60)

# Broadcasting allows operations on arrays of different shapes
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Array:\n{arr}")

# Add scalar (broadcasts to all elements)
result = arr + 10
print(f"\nArray + 10:\n{result}")

# Add 1D array to 2D array
row = np.array([10, 20, 30])
result = arr + row
print(f"\nArray + row {row}:\n{result}")

col = np.array([[10], [20], [30]])
result = arr + col
print(f"\nArray + column:\n{result}")

# ============================================================================
# 9. PRACTICAL ML EXAMPLE: Data Normalization
# ============================================================================

print("\n" + "=" * 60)
print("9. PRACTICAL ML EXAMPLE: DATA NORMALIZATION")
print("=" * 60)

# Simulate a dataset (rows=samples, columns=features)
np.random.seed(42)
dataset = np.random.randint(0, 100, size=(5, 3))
print(f"Original dataset:\n{dataset}")

# Min-Max Normalization (scale to [0, 1])
min_vals = dataset.min(axis=0)
max_vals = dataset.max(axis=0)
normalized = (dataset - min_vals) / (max_vals - min_vals)
print(f"\nMin-Max Normalized:\n{normalized}")

# Standardization (zero mean, unit variance)
mean = dataset.mean(axis=0)
std = dataset.std(axis=0)
standardized = (dataset - mean) / std
print(f"\nStandardized:\n{standardized}")
print(f"Mean: {standardized.mean(axis=0)}")
print(f"Std: {standardized.std(axis=0)}")

print("\n" + "=" * 60)
print("NUMPY FUNDAMENTALS COMPLETE!")
print("=" * 60)
