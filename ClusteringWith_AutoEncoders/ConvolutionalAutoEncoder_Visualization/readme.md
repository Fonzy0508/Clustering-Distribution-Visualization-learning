# Fashion-MNIST Embeddings with CAE & Clustering

Welcome! 😊 This project explores the Fashion-MNIST dataset using a Convolutional Autoencoder (CAE) and applies various dimensionality reduction techniques to the learned embeddings. After reducing dimensions, we cluster the data using different clustering algorithms. Let’s dive in! 🚀

## ✨ What’s Inside?

This project walks through:

- Training a **Convolutional Autoencoder (CAE)** on Fashion-MNIST 👕👟
- Extracting embeddings from the CAE
- Applying **dimensionality reduction** techniques:
  - **UMAP** 🗺️
  - **PCA** 📉
  - **t-SNE** 🔍
- Clustering the reduced data using:
  - **DBSCAN** 📌
  - **HDBSCAN** 📍
  - **K-Means** 🎯

## 🛠 Tools Used

This project was built using **Google Colab** and the following libraries:

- **TensorFlow Keras** - for the CAE model 🤖
- **Matplotlib** - for visualizing results 📊
- **NumPy** - for numerical computations 🔢
- **Scikit-learn** - for PCA, t-SNE, and clustering algorithms 🏗️
- **UMAP** - for dimensionality reduction 🚀

## 📂 How to Use

1. Open the notebook in **Google Colab**
2. Run the cells step by step to:
   - Load Fashion-MNIST
   - Train the CAE
   - Extract embeddings
   - Reduce dimensions
   - Apply clustering algorithms
3. Visualize and analyze the results 🎨

## 🎯 Goals

- Understand how CAE embeddings capture meaningful features
- Compare different dimensionality reduction methods
- Evaluate how clustering performs on reduced data
