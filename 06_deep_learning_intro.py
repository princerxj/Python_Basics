"""
Deep Learning Introduction with TensorFlow/Keras
=================================================

Introduction to deep learning using TensorFlow and Keras.
Covers neural networks, CNNs, and common deep learning patterns.
"""

import numpy as np
import matplotlib.pyplot as plt

# Note: TensorFlow might not be installed. This file shows the concepts.
# To run this code, install TensorFlow: pip install tensorflow

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not installed. Showing code examples only.")
    print("Install with: pip install tensorflow")

print("=" * 70)
print("DEEP LEARNING WITH TENSORFLOW/KERAS")
print("=" * 70)

if TENSORFLOW_AVAILABLE:
    print(f"TensorFlow version: {tf.__version__}")
else:
    print("TensorFlow not available - code examples only")

# ============================================================================
# 1. NEURAL NETWORK BASICS
# ============================================================================

print("\n" + "=" * 70)
print("1. BUILDING A SIMPLE NEURAL NETWORK")
print("=" * 70)

if TENSORFLOW_AVAILABLE:
    # Create a simple sequential model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(20,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = (X_train.sum(axis=1) > 0).astype(int)
    X_test = np.random.randn(200, 20)
    y_test = (X_test.sum(axis=1) > 0).astype(int)
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

else:
    print("""
Example code (requires TensorFlow):

model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
""")

# ============================================================================
# 2. ACTIVATION FUNCTIONS
# ============================================================================

print("\n" + "=" * 70)
print("2. ACTIVATION FUNCTIONS")
print("=" * 70)

print("""
Common activation functions:

1. ReLU (Rectified Linear Unit): max(0, x)
   - Most common for hidden layers
   - Fast computation
   - Helps with vanishing gradient

2. Sigmoid: 1 / (1 + e^(-x))
   - Output range: (0, 1)
   - Used for binary classification output

3. Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
   - Output range: (-1, 1)
   - Zero-centered

4. Softmax: e^(x_i) / sum(e^(x_j))
   - Used for multi-class classification output
   - Outputs probabilities that sum to 1

5. LeakyReLU: max(0.01*x, x)
   - Variant of ReLU that allows small negative values
""")

# ============================================================================
# 3. REGULARIZATION TECHNIQUES
# ============================================================================

print("\n" + "=" * 70)
print("3. REGULARIZATION TECHNIQUES")
print("=" * 70)

if TENSORFLOW_AVAILABLE:
    # Model with dropout and L2 regularization
    model_reg = Sequential([
        Dense(64, activation='relu', input_shape=(20,), 
              kernel_regularizer=keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(32, activation='relu',
              kernel_regularizer=keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    print("Model with regularization:")
    model_reg.summary()

else:
    print("""
Example code with regularization:

model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    Dropout(0.5),  # Dropout 50% of neurons
    Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
""")

print("""
Regularization techniques:

1. Dropout: Randomly drops neurons during training
   - Prevents overfitting
   - Typical values: 0.2 to 0.5

2. L1/L2 Regularization: Adds penalty to loss function
   - L1: Promotes sparsity
   - L2: Prevents large weights

3. Early Stopping: Stop training when validation loss stops improving

4. Batch Normalization: Normalizes layer inputs
   - Faster convergence
   - Reduces internal covariate shift
""")

# ============================================================================
# 4. CALLBACKS
# ============================================================================

print("\n" + "=" * 70)
print("4. CALLBACKS")
print("=" * 70)

print("""
Callbacks are functions called during training:

1. EarlyStopping: Stop training when metric stops improving
2. ModelCheckpoint: Save model at intervals
3. ReduceLROnPlateau: Reduce learning rate when metric plateaus
4. TensorBoard: Visualize training progress
5. Custom callbacks: Define your own behavior

Example:

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)

history = model.fit(
    X_train, y_train,
    callbacks=[early_stop, checkpoint],
    validation_split=0.2
)
""")

# ============================================================================
# 5. CONVOLUTIONAL NEURAL NETWORKS (CNN)
# ============================================================================

print("\n" + "=" * 70)
print("5. CONVOLUTIONAL NEURAL NETWORKS (CNN)")
print("=" * 70)

if TENSORFLOW_AVAILABLE:
    # Build CNN for image classification
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    print("CNN Architecture:")
    cnn_model.summary()

else:
    print("""
CNN Example for image classification:

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
""")

print("""
CNN Layers:

1. Conv2D: Convolutional layer
   - Extracts features using filters
   - Parameters: filters, kernel_size, activation

2. MaxPooling2D: Downsampling layer
   - Reduces spatial dimensions
   - Retains important features

3. Flatten: Converts 2D to 1D
   - Prepares for dense layers

4. Dense: Fully connected layer
   - Traditional neural network layer
""")

# ============================================================================
# 6. TRAINING ON MNIST DATASET
# ============================================================================

print("\n" + "=" * 70)
print("6. EXAMPLE: MNIST DIGIT CLASSIFICATION")
print("=" * 70)

if TENSORFLOW_AVAILABLE:
    # Load MNIST dataset
    (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()
    
    # Preprocess
    X_train_mnist = X_train_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255
    X_test_mnist = X_test_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train_mnist = keras.utils.to_categorical(y_train_mnist, 10)
    y_test_mnist = keras.utils.to_categorical(y_test_mnist, 10)
    
    print(f"Training data shape: {X_train_mnist.shape}")
    print(f"Test data shape: {X_test_mnist.shape}")
    
    # Build and compile model
    mnist_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    mnist_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("\nTraining on MNIST...")
    history_mnist = mnist_model.fit(
        X_train_mnist, y_train_mnist,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = mnist_model.evaluate(X_test_mnist, y_test_mnist, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

else:
    print("""
Example: Training on MNIST dataset

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
""")

# ============================================================================
# 7. OPTIMIZERS
# ============================================================================

print("\n" + "=" * 70)
print("7. OPTIMIZERS")
print("=" * 70)

print("""
Common optimizers:

1. SGD (Stochastic Gradient Descent)
   - Basic optimizer
   - Can add momentum
   
   optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

2. Adam (Adaptive Moment Estimation)
   - Most popular
   - Adapts learning rate for each parameter
   - Good default choice
   
   optimizer = keras.optimizers.Adam(learning_rate=0.001)

3. RMSprop
   - Good for recurrent neural networks
   
   optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

4. AdaGrad
   - Adapts learning rate based on parameters
   
   optimizer = keras.optimizers.Adagrad(learning_rate=0.01)
""")

# ============================================================================
# 8. SAVING AND LOADING MODELS
# ============================================================================

print("\n" + "=" * 70)
print("8. SAVING AND LOADING MODELS")
print("=" * 70)

if TENSORFLOW_AVAILABLE:
    # Save entire model
    model.save('my_model.h5')
    print("Model saved to 'my_model.h5'")
    
    # Load model
    loaded_model = keras.models.load_model('my_model.h5')
    print("Model loaded successfully")
    
    # Save only weights
    model.save_weights('model_weights.h5')
    print("Weights saved to 'model_weights.h5'")

else:
    print("""
Saving and loading models:

# Save entire model
model.save('my_model.h5')

# Load model
loaded_model = keras.models.load_model('my_model.h5')

# Save only weights
model.save_weights('model_weights.h5')

# Load weights
model.load_weights('model_weights.h5')

# Save in TensorFlow SavedModel format
model.save('my_model')

# Load SavedModel
loaded_model = keras.models.load_model('my_model')
""")

# ============================================================================
# 9. TRANSFER LEARNING
# ============================================================================

print("\n" + "=" * 70)
print("9. TRANSFER LEARNING")
print("=" * 70)

print("""
Transfer Learning: Use pre-trained models on new tasks

Example with VGG16:

from tensorflow.keras.applications import VGG16

# Load pre-trained VGG16 (without top classification layer)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Add custom classification layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

Popular pre-trained models:
- VGG16/VGG19: Image classification
- ResNet50: Image classification
- InceptionV3: Image classification
- MobileNet: Lightweight for mobile devices
- BERT: Natural language processing
""")

# ============================================================================
# 10. BEST PRACTICES
# ============================================================================

print("\n" + "=" * 70)
print("10. DEEP LEARNING BEST PRACTICES")
print("=" * 70)

print("""
1. DATA PREPARATION:
   - Normalize/standardize inputs
   - Split into train/validation/test sets
   - Use data augmentation for images
   
2. MODEL ARCHITECTURE:
   - Start simple, add complexity if needed
   - Use ReLU for hidden layers
   - Use appropriate output activation (sigmoid, softmax)
   
3. REGULARIZATION:
   - Use dropout (0.2-0.5)
   - Apply L2 regularization
   - Use early stopping
   - Use batch normalization
   
4. TRAINING:
   - Use appropriate batch size (32, 64, 128)
   - Monitor both training and validation metrics
   - Use callbacks (early stopping, model checkpoint)
   - Try different learning rates
   
5. EVALUATION:
   - Always use separate test set
   - Use cross-validation for small datasets
   - Monitor for overfitting (train vs val loss)
   
6. OPTIMIZATION:
   - Start with Adam optimizer
   - Adjust learning rate if needed
   - Use learning rate scheduling
   
7. DEBUGGING:
   - Check data shapes and types
   - Verify data preprocessing
   - Start with small dataset to test pipeline
   - Monitor loss curves
""")

print("\n" + "=" * 70)
print("DEEP LEARNING FUNDAMENTALS COMPLETE!")
print("=" * 70)
