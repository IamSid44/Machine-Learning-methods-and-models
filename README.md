# Statistical Methods in AI

A comprehensive collection of solutions to assignments from the **Statistical Methods in AI (SMAI)** course. This repository contains implementations of fundamental to advanced machine learning algorithms, organized by topic with detailed problem statements, approaches, and results.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Problem Descriptions](#problem-descriptions)
  - [Regression](#regression)
  - [Clustering](#clustering)
  - [Classification](#classification)
  - [Neural Networks](#neural-networks)
  - [Generative Models](#generative-models)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)

---

## Overview

This repository serves as a portfolio of statistical and machine learning implementations covering:
- **Classical ML**: Regression, clustering, tree-based models
- **Deep Learning**: MLPs, CNNs, RNNs, and autoencoders
- **Unsupervised Learning**: K-Means, GMM, dimensionality reduction
- **Generative Models**: Variational autoencoders and diffusion models

Most algorithms are implemented from scratch without relying on high-level abstractions.

---

## Repository Structure

```
Machine-Learning-methods-and-models/
├── regression/                    # Regression
├── clustering/                    # Unsupervised clustering methods
├── classification/                # Classification and decision trees
├── neural-networks/               # Deep learning and MLPs
├── generative-models/             # Generative modeling approaches
├── utilities/                     # Foundational concepts and data analysis 
└── README.md               
```

---

## Problem Descriptions

### Regression

#### 01 - Linear Regression with Regularization

**Problem Statement:**
Predict student GPA using feature data with polynomial regression. Implement regularization techniques to prevent overfitting and analyze feature importance.

**Tasks:**
- Fit polynomial regression models (degrees 1-6) without regularization
- Compare L1 and L2 regularization across different strengths (alpha)
- Use validation set MSE to select optimal hyperparameters
- Analyze which features are most predictive of GPA
- Compare feature importance between L1 and L2 regularization

**Approach:**
- Implement polynomial feature expansion
- Use MSE loss with L1/L2 penalties
- Hyperparameter tuning via validation set
- Feature importance analysis through non-zero weights

**Key Results:**
- Identified optimal polynomial degree and regularization strength
- L1 regularization provides feature selection; L2 provides smooth coefficients
- Validation MSE comparison shows regularization impact on generalization

**Notebook:** `regression/01_linear_regression_regularization.ipynb`

---

### Clustering

#### 02 - K-Means Clustering

**Problem Statement:**
Implement K-Means clustering from scratch and determine the optimal number of clusters using multiple evaluation metrics.

**Tasks:**
- Implement custom K-Means class with `fit()`, `predict()`, and `getCost()` methods
- Determine optimal k using Elbow Method (WCSS analysis)
- Evaluate cluster quality using Silhouette Score
- Compare custom implementation with sklearn's K-Means
- Visualize clustering results

**Approach:**
- Random initialization of centroids
- Iterative expectation-maximization until convergence
- WCSS (Within-Cluster Sum of Squares) calculation
- Elbow method and Silhouette score for k-selection

**Key Results:**
- Successfully identified optimal cluster count
- Custom implementation matches sklearn performance
- Visualized cluster quality metrics

**Notebook:** `clustering/01_kmeans_clustering.ipynb`

---

#### 03 - Gaussian Mixture Models (GMM)

**Problem Statement:**
Implement Gaussian Mixture Models using the EM algorithm to perform probabilistic clustering with soft cluster assignments.

**Tasks:**
- Implement custom GMM class with `fit()`, `getMembership()`, and `getLikelihood()` methods
- Apply EM algorithm to determine Gaussian parameters (mean, covariance, weights)
- Determine optimal clusters using BIC (Bayesian Information Criterion)
- Evaluate using Silhouette Method
- Visualize likelihood convergence across iterations

**Approach:**
- E-step: Compute responsibility (membership) of each point to each Gaussian
- M-step: Update Gaussian parameters based on responsibilities
- BIC for model selection
- Likelihood tracking to visualize convergence

**Key Results:**
- EM algorithm converged successfully
- BIC provided reliable cluster selection
- Likelihood improvement visualized across iterations

**Notebook:** `clustering/02_gaussian_mixture_models.ipynb`

---

#### 04 - Image Segmentation using GMM

**Problem Statement:**
Apply custom GMM implementation to perform color-based image segmentation on satellite imagery. Create visualization videos showing EM algorithm convergence.

**Tasks:**
- Load satellite images and preprocess pixel data
- Fit GMM with k=3 components (Land, Water, Vegetation)
- Segment images based on learned Gaussians
- Create videos showing EM convergence frame-by-frame
- Generate segmentation with original and labeled outputs

**Approach:**
- Flatten image pixels into feature vectors
- Apply GMM for color-based clustering
- Assign each pixel to most likely Gaussian
- Visualize convergence and final segmentation

**Key Results:**
- Clear separation of land, water, and vegetation regions
- Convergence visualization shows EM progress
- High-quality segmentation maps

**Notebook:** `clustering/03_image_segmentation_gmm.ipynb`

---

#### 05 - PCA + Classification

**Problem Statement:**
Implement PCA from scratch and evaluate feature-space compression through downstream classification performance.

**Notebook:** `clustering/04_pca_classification.ipynb`

---

### Classification

#### 01 - Multi-Task CNN (Fashion-MNIST)

**Problem Statement:**
Build a CNN that jointly performs classification and regression on Fashion-MNIST dataset. Balance two tasks through joint loss optimization.

**Tasks:**
- Implement multi-task CNN with shared feature layers
- Task 1: Classify 10 clothing types (Cross-Entropy Loss)
- Task 2: Predict normalized pixel intensity (MSE Loss)
- Optimize joint loss: L = λ₁L_CE + λ₂L_MSE with varying weights
- Visualize feature maps at different layers
- Analyze task interaction through hyperparameter sweeps

**Approach:**
- Shared convolutional backbone
- Task-specific output heads
- Joint loss with learnable/fixed weight ratios
- Hyperparameter tuning using Weights & Biases

**Key Results:**
- Both tasks improve with shared representations
- Balanced loss weighting achieves best overall performance
- Feature visualizations show task-relevant patterns

**Notebook:** `classification/01_multitask_cnn_fmnist.ipynb`

---

#### 02 - Image Colorization (CIFAR-10)

**Problem Statement:**
Build an encoder-decoder CNN to perform automatic image colorization. Treat colorization as a classification problem mapping pixels to learned color clusters.

**Tasks:**
- Implement encoder for feature extraction from grayscale images
- Design decoder with upsampling layers (ConvTranspose2d)
- Define 24 representative color clusters as classification targets
- Perform extensive hyperparameter tuning (learning rate, batch size, filters)
- Use Weights & Biases for sweep tracking
- Visualize colorized outputs and compare with originals

**Approach:**
- Grayscale input → feature extraction → color cluster prediction
- Learned upsampling for spatial resolution recovery
- Cross-entropy loss for color classification
- Hyperparameter sweeps for optimization

**Key Results:**
- Successfully learned plausible color assignments
- Identified optimal architecture and hyperparameters
- Generated colored outputs with semantic consistency

**Notebook:** `classification/02_image_colorization_cifar10.ipynb`

---

#### 03 - Decision Trees for Text Classification

**Problem Statement:**
Implement decision tree classifiers for Amazon product review sentiment analysis. Compare tree-from-scratch vs sklearn with various hyperparameters.

**Tasks:**
- Implement decision tree from scratch with binary split selection
- Train on bag-of-words representation (7729 features)
- Perform grid search over max_depth and min_samples_leaf
- Compare custom implementation with sklearn trees
- Visualize tree structure and decision boundaries
- Optimize using balanced accuracy metric

**Approach:**
- Binary split selection via information gain/Gini impurity
- Recursive tree construction with stopping criteria
- Grid search for hyperparameter tuning
- Balanced accuracy for imbalanced datasets

**Key Results:**
- Custom tree implementation matches sklearn performance
- Identified optimal depth and min_samples constraints
- Successfully classified sentiment from text features

**Notebook:** `classification/03_decision_trees_text.ipynb`

---

### Neural Networks

#### 01 - Border Prediction MLP (Baarle-Nassau)

**Problem Statement:**
Implement a complete MLP from scratch to model the complex Belgium-Netherlands border at Baarle-Nassau. The border has an intricate pattern requiring a non-linear classifier.

**Tasks:**
- Implement Linear layer class with forward/backward passes
- Implement activation functions: ReLU, Tanh, Sigmoid, Identity
- Build complete MLP with gradient accumulation and early stopping
- Train to predict country (Netherlands=0, Belgium=1) from pixel coordinates
- Achieve ~91% accuracy through architecture optimization
- Visualize learned decision boundaries

**Approach:**
- Core components: Linear layers, activations, loss functions
- Forward pass: Sequential layer execution
- Backward pass: Gradient computation via chain rule
- Training loop with batch processing and early stopping

**Key Results:**
- Achieved 91% accuracy on complex border prediction
- Learned non-linear decision boundary
- Decision boundary visualization shows learned border complexity

**Notebook:** `neural-networks/01_border_prediction_mlp.ipynb`

---

#### 02-06 - Additional Neural Network Problems

**Problem Statements:**
Implementation of advanced neural network architectures and training techniques.

**Topics Include:**
- Feature mapping techniques (raw pixels, Taylor expansion, Fourier features)
- Image reconstruction using coordinate-based networks
- Autoencoders for dimensionality reduction and reconstruction
- Sequence modeling with linear/MLP/RNN recurrence predictors
- Evaluation on blurred images with varying Gaussian blur levels

**Notebooks:**
- `neural-networks/02_*.ipynb`
- `neural-networks/03_*.ipynb`
- `neural-networks/04_*.ipynb`
- `neural-networks/05_*.ipynb`
- `neural-networks/06_*.ipynb`

---

### Generative Models

This section contains dedicated generative model notebooks.

#### 01 - Variational Autoencoders

**Notebook:** `generative-models/01_variational_autoencoders.ipynb`

#### 02 - Diffusion Models

**Notebook:** `generative-models/02_diffusion_models.ipynb`

---

### Classical Techniques

#### 01 - K-Nearest Neighbors Classifier

**Problem:** Implement KNN from scratch for classification with custom feature transformation.

**Notebook:** `utilities/01_knn_classifier.ipynb`

---

#### 02 - Polynomial Regression Analysis

**Problem:** Implement and analyze polynomial regression with various degrees and regularization.

**Notebook:** `utilities/02_polynomial_regression.ipynb`

---

#### 03 - Data Analysis and Visualization

**Problem:** Explore and analyze dataset characteristics, distributions, and relationships.

**Notebook:** `utilities/03_data_analysis.ipynb`

---

## Setup Instructions

- Dependencies specified in `requirements.txt`

```bash
# Install uv 
pip install uv

# Install dependencies into system Python 
uv pip install --system -r requirements.txt
```

Navigate to the relevant notebook and run cells sequentially. Each notebook is self-contained and includes:

1. **Problem Statement** - Clear description of the task
2. **Implementation** - Code with explanations
3. **Experiments** - Hyperparameter tuning and sensitivity analysis
4. **Visualization** - Plots, tables, and insights
5. **Results & Analysis** - Key findings and conclusions

### Key Implementation Highlights

- **From-Scratch Implementations**: K-Means, GMM, Neural Networks, Decision Trees
- **Experimentation**: Comprehensive hyperparameter tuning and ablation studies
- **Visualization**: Rich plots for understanding model behavior

- **Data Files**: Some notebooks require external datasets (MNIST, CIFAR-10, FMNIST). These are downloaded automatically on first run.
- **GPU Support**: PyTorch operations can leverage GPU if available (CUDA-enabled devices).
- **Reproducibility**: Most notebooks use fixed random seeds for consistent results.
- **Dependencies**: See `requirements.txt` for complete list of packages and versions.