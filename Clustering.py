import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

## Loading Data sets from sklearn
from sklearn.datasets import load_digits

print("All packages imported successfully!")

##Assigning the data to a variable
digits = load_digits()

## Extracting feautures and labes from the digit dataset
x = digits.data
y = digits.target

## Checks the shape of the data
print(f"Data features {x.shape}") 
print(f"Target shape {y.shape}")

## checks all the labels
print(f"Unique labels: {set(map(int, y))}")



# fig, axes = plt.subplots(1, 4, figsize=(5, 3))
# for i, ax in enumerate(axes):
#     ax.imshow(digits.images[i], cmap='viridis')
#     ax.set_title(f"Label: {digits.target[i]}")
#     ax.axis('off')

# plt.show()

## Initialize PCA (reduce to 2 dimensions)
pca = PCA(n_components=2)

## Fit PCA to the data and transform it
X_pca = pca.fit_transform(x)

## Print the new shape
print(f"Original shape: {x.shape}")   # Should print (1797, 64)
print(f"PCA-reduced shape: {X_pca.shape}")  # Should print (1797, 2)

## Visualize the PCA result
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Spectral', alpha=0.7)
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('PCA - Digits Dataset')
# plt.colorbar(scatter, label='Digit Label')
#plt.show()

## Import K-Means
from sklearn.cluster import KMeans, DBSCAN

## initialize K-means to 10 clusters(0-9)
kmeans =KMeans(n_clusters=10,random_state=42)

## fit kmeand to pca reduced data
kmeans_labels = kmeans.fit_predict(X_pca)

## Plot the K-Means Clustering result
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering on PCA-Reduced Digits Data')
plt.colorbar(scatter, label='Cluster')
plt.show()

## import score checkers
from sklearn.metrics import silhouette_score

## Calculate Silhouette Score
score = silhouette_score(X_pca, kmeans_labels)
print(f"Silhouette Score for K-Means Clustering: {score:.2f}")

# Initialize DBSCAN
dbscan = DBSCAN(eps=2.0, min_samples=5)

# Fit DBSCAN to PCA-reduced data
dbscan_labels = dbscan.fit_predict(X_pca)

# Count unique clusters (DBSCAN labels noise points as -1)
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")

# Plot DBSCAN clusters
plt.figure(figsize=(8, 6))

# Assign colors: -1 (noise) will be plotted in black
unique_labels = set(dbscan_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for label, col in zip(unique_labels, colors):
    if label == -1:
        # Black for noise
        col = [0, 0, 0, 1]

    class_member_mask = (dbscan_labels == label)
    xy = X_pca[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title(f'DBSCAN Clustering (Estimated Clusters: {n_clusters})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Apply Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=10, linkage='ward')
agg_labels = agg_clustering.fit_predict(X_pca)

# Calculate Silhouette Score
score = silhouette_score(X_pca, agg_labels)
print(f"Silhouette Score for Hierarchical Clustering: {score:.2f}")

# Plot Hierarchical Clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='Spectral', s=30)
plt.title("Hierarchical Clustering on PCA-Reduced Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster')
plt.show()

# Generate linkage matrix
linked = linkage(X_pca, method='ward')

# Plot Dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level', p=5)  # Show only top 5 levels
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
