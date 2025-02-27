# MNIST Embeddings & Clustering

Welcome! 😊 This project explores the classic **MNIST** dataset (digits 0-9) and applies various **dimensionality reduction** techniques to the raw image data. After reducing dimensions, we cluster the data using different clustering algorithms. Let’s dive in! 🚀

## ✨ What’s Inside?

This project walks through:

- Loading and preprocessing the **MNIST dataset** 0️⃣1️⃣2️⃣...9️⃣
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

- **Matplotlib** - for visualizing results 📊
- **NumPy** - for numerical computations 🔢
- **Scikit-learn** - for PCA, t-SNE, and clustering algorithms 🏗️
- **UMAP** - for dimensionality reduction 🚀

## 📂 How to Use

1. Open the notebook in **Google Colab**
2. Run the cells step by step to:
   - Load the MNIST dataset
   - Preprocess the data
   - Reduce dimensions
   - Apply clustering algorithms
3. Visualize and analyze the results 🎨

## 🎯 Goals

- Understand how dimensionality reduction techniques transform MNIST data
- Compare different dimensionality reduction methods
- Evaluate how clustering performs on reduced data

## 📝 Notes

- Experiment with different numbers of PCA components or t-SNE perplexity values.
- UMAP often preserves local structures better than PCA/t-SNE.
- The clustering results depend on the choice of dimensionality reduction technique.
