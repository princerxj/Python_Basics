"""
Python Basics for AI/ML
========================

This module covers fundamental Python concepts essential for AI and Machine Learning.
"""

# ============================================================================
# 1. VARIABLES AND DATA TYPES
# ============================================================================

# Basic data types
integer_var = 42
float_var = 3.14
string_var = "Machine Learning"
boolean_var = True

print("=" * 50)
print("DATA TYPES")
print("=" * 50)
print(f"Integer: {integer_var}, Type: {type(integer_var)}")
print(f"Float: {float_var}, Type: {type(float_var)}")
print(f"String: {string_var}, Type: {type(string_var)}")
print(f"Boolean: {boolean_var}, Type: {type(boolean_var)}")

# ============================================================================
# 2. DATA STRUCTURES
# ============================================================================

print("\n" + "=" * 50)
print("DATA STRUCTURES")
print("=" * 50)

# Lists - ordered, mutable collections
features = ['age', 'height', 'weight', 'income']
print(f"Features list: {features}")
print(f"First feature: {features[0]}")
print(f"Last feature: {features[-1]}")

# Tuples - ordered, immutable collections
model_params = (0.01, 100, 0.9)  # learning_rate, epochs, momentum
print(f"Model parameters: {model_params}")

# Dictionaries - key-value pairs (like JSON)
model_config = {
    'learning_rate': 0.01,
    'batch_size': 32,
    'epochs': 100,
    'optimizer': 'adam'
}
print(f"Model config: {model_config}")
print(f"Learning rate: {model_config['learning_rate']}")

# Sets - unordered collections of unique elements
unique_labels = {0, 1, 1, 0, 1, 2}  # duplicates removed
print(f"Unique labels: {unique_labels}")

# ============================================================================
# 3. CONTROL FLOW
# ============================================================================

print("\n" + "=" * 50)
print("CONTROL FLOW")
print("=" * 50)

# If-else statements
accuracy = 0.95
if accuracy > 0.9:
    print(f"Excellent model! Accuracy: {accuracy}")
elif accuracy > 0.7:
    print(f"Good model! Accuracy: {accuracy}")
else:
    print(f"Model needs improvement. Accuracy: {accuracy}")

# For loops - iterate over sequences
print("\nIterating over features:")
for feature in features:
    print(f"  - {feature}")

# While loops
epoch = 0
max_epochs = 5
print(f"\nTraining simulation:")
while epoch < max_epochs:
    print(f"  Epoch {epoch + 1}/{max_epochs}")
    epoch += 1

# ============================================================================
# 4. LIST COMPREHENSIONS (Very useful in ML)
# ============================================================================

print("\n" + "=" * 50)
print("LIST COMPREHENSIONS")
print("=" * 50)

# Create a list of squared values
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(f"Original: {numbers}")
print(f"Squared: {squared}")

# Filter even numbers
even_numbers = [x for x in numbers if x % 2 == 0]
print(f"Even numbers: {even_numbers}")

# Normalize values (common in ML preprocessing)
max_val = max(numbers)
normalized = [x / max_val for x in numbers]
print(f"Normalized: {normalized}")

# ============================================================================
# 5. FUNCTIONS
# ============================================================================

print("\n" + "=" * 50)
print("FUNCTIONS")
print("=" * 50)

def calculate_accuracy(correct, total):
    """Calculate accuracy percentage."""
    return (correct / total) * 100

def train_test_split(data, test_ratio=0.2):
    """Split data into train and test sets."""
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]

# Using functions
acc = calculate_accuracy(95, 100)
print(f"Accuracy: {acc}%")

dataset = list(range(100))
train, test = train_test_split(dataset)
print(f"Train size: {len(train)}, Test size: {len(test)}")

# Lambda functions (anonymous functions)
square = lambda x: x**2
print(f"Square of 5: {square(5)}")

# ============================================================================
# 6. CLASSES AND OBJECTS (Object-Oriented Programming)
# ============================================================================

print("\n" + "=" * 50)
print("CLASSES AND OBJECTS")
print("=" * 50)

class MLModel:
    """A simple machine learning model class."""
    
    def __init__(self, model_name, learning_rate=0.01):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.is_trained = False
        self.accuracy = 0.0
    
    def train(self, epochs):
        """Simulate training the model."""
        print(f"Training {self.model_name} for {epochs} epochs...")
        self.is_trained = True
        # Simulate improving accuracy
        self.accuracy = min(0.95, 0.5 + epochs * 0.05)
        print(f"Training complete! Accuracy: {self.accuracy:.2f}")
    
    def predict(self, data):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")
        return f"Predictions for {len(data)} samples"

# Create and use a model
model = MLModel("Neural Network", learning_rate=0.001)
model.train(epochs=10)
result = model.predict([1, 2, 3, 4, 5])
print(result)

# ============================================================================
# 7. FILE I/O (Important for loading datasets)
# ============================================================================

print("\n" + "=" * 50)
print("FILE I/O")
print("=" * 50)

# Writing to a file
with open('sample_data.txt', 'w') as f:
    f.write("feature1,feature2,label\n")
    f.write("1.0,2.0,0\n")
    f.write("1.5,1.8,1\n")
    f.write("3.0,4.0,0\n")
print("Data written to sample_data.txt")

# Reading from a file
with open('sample_data.txt', 'r') as f:
    content = f.readlines()
    print("\nReading from file:")
    for line in content:
        print(f"  {line.strip()}")

# ============================================================================
# 8. EXCEPTION HANDLING
# ============================================================================

print("\n" + "=" * 50)
print("EXCEPTION HANDLING")
print("=" * 50)

def safe_divide(a, b):
    """Safely divide two numbers."""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    except TypeError:
        print("Error: Invalid input types!")
        return None
    finally:
        print("Division operation completed.")

print(f"10 / 2 = {safe_divide(10, 2)}")
print(f"10 / 0 = {safe_divide(10, 0)}")

print("\n" + "=" * 50)
print("PYTHON BASICS COMPLETE!")
print("=" * 50)
