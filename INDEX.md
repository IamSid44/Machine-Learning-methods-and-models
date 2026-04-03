# SMAI Solutions Index

This index is the fastest way to navigate the repository by topic and notebook.

## Quick Navigation

- [Classification](classification/README.md)
- [Clustering](clustering/README.md)
- [Generative Models](generative-models/README.md)
- [Neural Networks](neural-networks/README.md)
- [Regression](regression/README.md)
- [Utilities](utilities/README.md)

## Notebook Map

### Classification

- [01_multitask_cnn_fmnist.ipynb](classification/01_multitask_cnn_fmnist.ipynb): Multi-task CNN on Fashion-MNIST (classification + regression).
- [02_image_colorization_cifar10.ipynb](classification/02_image_colorization_cifar10.ipynb): Image colorization as a structured prediction/classification task.
- [03_decision_trees_text.ipynb](classification/03_decision_trees_text.ipynb): Decision trees for sentiment classification on text features.

### Clustering

- [01_kmeans_clustering.ipynb](clustering/01_kmeans_clustering.ipynb): Custom K-Means with elbow and silhouette analysis.
- [02_gaussian_mixture_models.ipynb](clustering/02_gaussian_mixture_models.ipynb): Custom GMM with EM and BIC-based model selection.
- [03_image_segmentation_gmm.ipynb](clustering/03_image_segmentation_gmm.ipynb): GMM-based image segmentation and convergence visualization.

### Generative Models

- [01_diffusion_models.ipynb](generative-models/01_diffusion_models.ipynb): DDPM-style diffusion model tutorial and sampling workflow.

### Neural Networks

- [01_border_prediction_mlp.ipynb](neural-networks/01_border_prediction_mlp.ipynb): MLP from scratch for complex border prediction.
- [02_feature_mapping_image_reconstruction.ipynb](neural-networks/02_feature_mapping_image_reconstruction.ipynb): Feature mappings for coordinate-based image reconstruction.
- [03_autoencoders_anomaly_detection.ipynb](neural-networks/03_autoencoders_anomaly_detection.ipynb): Autoencoder-based anomaly detection.
- [04_advanced_architectures.ipynb](neural-networks/04_advanced_architectures.ipynb): Extended architecture experiments and comparative analysis.

### Regression

- [01_linear_regression_regularization.ipynb](regression/01_linear_regression_regularization.ipynb): Polynomial regression with L1/L2 regularization analysis.

### Utilities

- [01_knn_classifier.ipynb](utilities/01_knn_classifier.ipynb): KNN pipeline and sampling analysis utilities.
- [02_polynomial_regression.ipynb](utilities/02_polynomial_regression.ipynb): Compact polynomial regression experiments.
- [03_data_analysis.ipynb](utilities/03_data_analysis.ipynb): Exploratory data analysis and supporting plots.

## Execution Notes

- Heavy deep learning notebooks are intentionally not fully retrained in this pass.
- Selective light execution was run for:
  - [utilities/01_knn_classifier.ipynb](utilities/01_knn_classifier.ipynb)
  - [utilities/02_polynomial_regression.ipynb](utilities/02_polynomial_regression.ipynb)
  - [utilities/03_data_analysis.ipynb](utilities/03_data_analysis.ipynb)
  - [regression/01_linear_regression_regularization.ipynb](regression/01_linear_regression_regularization.ipynb)
- Saved outputs and result snapshots are used where retraining is expensive.
