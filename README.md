# Python for AI and Machine Learning

A comprehensive learning resource for Python fundamentals and essential libraries for AI/ML development.

## 📚 Contents

This repository contains practical, hands-on examples covering the essential topics for AI and Machine Learning with Python:

### 1. Python Basics (`01_python_basics.py`)
- Variables and data types
- Data structures (lists, tuples, dictionaries, sets)
- Control flow (if/else, loops)
- List comprehensions
- Functions and lambda expressions
- Classes and object-oriented programming
- File I/O operations
- Exception handling

### 2. NumPy Fundamentals (`02_numpy_fundamentals.py`)
- Creating and manipulating arrays
- Array attributes and indexing
- Mathematical operations
- Aggregation functions
- Array manipulation (reshape, transpose, concatenate)
- Linear algebra operations
- Broadcasting
- Practical example: Data normalization

### 3. Pandas Fundamentals (`03_pandas_fundamentals.py`)
- Series and DataFrames
- Data selection and filtering
- Handling missing data
- GroupBy operations
- Data merging and joining
- Reading/writing data (CSV, Excel)
- Feature engineering
- Data preparation for ML

### 4. Data Visualization with Matplotlib (`04_matplotlib_visualization.py`)
- Line plots
- Scatter plots
- Bar plots and histograms
- Box plots and heatmaps
- Subplots and layouts
- ML-specific visualizations:
  - Training history plots
  - Confusion matrices
  - Feature importance charts
  - Decision boundaries

### 5. Machine Learning with Scikit-learn (`05_scikit_learn_ml.py`)
- Data preparation and preprocessing
- Train-test split and cross-validation
- Classification algorithms:
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
- Regression with Linear Regression
- Clustering with K-Means
- Model evaluation metrics
- Model comparison and selection
- Saving/loading models
- Complete ML pipelines

### 6. Deep Learning Introduction (`06_deep_learning_intro.py`)
- Neural network basics
- Activation functions
- Regularization techniques (Dropout, L1/L2)
- Training callbacks
- Convolutional Neural Networks (CNNs)
- MNIST digit classification example
- Optimizers (SGD, Adam, RMSprop)
- Model saving and loading
- Transfer learning concepts
- Best practices

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/princerxj/Python_Basics.git
cd Python_Basics
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Examples

Each file can be run independently:

```bash
# Python basics
python 01_python_basics.py

# NumPy fundamentals
python 02_numpy_fundamentals.py

# Pandas fundamentals
python 03_pandas_fundamentals.py

# Matplotlib visualization
python 04_matplotlib_visualization.py

# Scikit-learn ML
python 05_scikit_learn_ml.py

# Deep learning (requires TensorFlow)
python 06_deep_learning_intro.py
```

## 📦 Dependencies

Core libraries:
- **NumPy**: Numerical computing with arrays and matrices
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning algorithms and tools
- **TensorFlow/Keras**: Deep learning framework (optional)

See `requirements.txt` for specific versions.

## 📖 Learning Path

**Recommended order for beginners:**

1. **Start with Python Basics** - Understand the fundamentals
2. **Learn NumPy** - Master array operations
3. **Study Pandas** - Data manipulation skills
4. **Practice Matplotlib** - Visualization techniques
5. **Explore Scikit-learn** - Traditional ML algorithms
6. **Dive into Deep Learning** - Neural networks and advanced topics

## 💡 Key Concepts

### Data Science Workflow
1. **Data Collection**: Gather data from various sources
2. **Data Cleaning**: Handle missing values, outliers
3. **Data Exploration**: Visualize and understand patterns
4. **Feature Engineering**: Create meaningful features
5. **Model Selection**: Choose appropriate algorithms
6. **Model Training**: Fit models to training data
7. **Model Evaluation**: Assess performance on test data
8. **Model Deployment**: Use model in production

### Machine Learning Types
- **Supervised Learning**: Classification, Regression
- **Unsupervised Learning**: Clustering, Dimensionality Reduction
- **Deep Learning**: Neural Networks, CNNs, RNNs

## 🎯 Practical Applications

The skills covered in this repository can be applied to:
- Image classification and object detection
- Natural language processing
- Time series forecasting
- Recommendation systems
- Fraud detection
- Customer segmentation
- Predictive maintenance
- And many more!

## 🔧 Common Issues

### Import Errors
If you encounter import errors, ensure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

### TensorFlow Installation
For deep learning examples, TensorFlow may require specific setup:
```bash
# TensorFlow includes GPU support automatically when CUDA is available
pip install tensorflow
```

**Note for GPU users**: To use GPU acceleration, you need:
- NVIDIA GPU with CUDA compute capability 3.5 or higher
- CUDA Toolkit (compatible version with your TensorFlow)
- cuDNN library
- Check [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu) for detailed setup

## 📝 Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new examples
- Improve documentation
- Add more advanced topics

## 📄 License

This project is open source and available for educational purposes.

## ✨ Tips for Success

1. **Practice Regularly**: Run and modify the examples
2. **Experiment**: Change parameters and observe results
3. **Read Documentation**: Refer to official docs for deeper understanding
4. **Build Projects**: Apply concepts to real-world problems
5. **Join Communities**: Engage with other learners
6. **Stay Updated**: AI/ML field evolves rapidly

---

**Happy Learning! 🎓**

For questions or feedback, please open an issue in this repository.