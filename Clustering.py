import numpy as np
import pandas as pd
import umap 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

## Loading Data sets from sklearn
from sklearn.datasets import load_digits

## import score checkers
from sklearn.metrics import silhouette_score

print("All packages imported successfully!")

##Assigning the dataset to a variable
digits = load_digits()

## Extracting feautures and labes from the digit dataset
x = digits.data
y = digits.target

## Checks the shape of the data
print(f"Data features {x.shape}") 
print(f"Target shape {y.shape}")

## checks all the labels
print(f"Unique labels: {set(map(int, y))}")

def Init_UMAP():
    # Initialize UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)

    # Apply UMAP on the digits data
    X_umap = reducer.fit_transform(x)

    # Visualize UMAP output
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral', s=10)
    plt.title('UMAP Projection - Digits Dataset')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.colorbar(scatter, label='Digit Label')
    plt.show()
    return X_umap,"UMAP"

## FUNCTIONS FOR DIMENSION REDUCTION
def Init_PCA(data, isShowPlot = False):
    ## Initialize PCA (reduce to 2 dimensions)
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(data)
    
    if isShowPlot : 
        ## Visualize the PCA result
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='Spectral', alpha=0.7)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('PCA - Digits Dataset')
        plt.colorbar(scatter, label='Digit Label')
        plt.show()
    return x_pca,"PSA"

def Init_tSNE(data, labels, n_components=2, perplexity=30, learning_rate=200, isPCAEnabled=False):
    """
    Applies t-SNE to reduce dimensions and visualize the data.
    
    Parameters:
        data: High-dimensional data (e.g., 64D digits dataset)
        labels: True labels for coloring the plot
        n_components: Target dimension (2D or 3D)
        perplexity: Balances attention between local and global aspects
        learning_rate: Controls the step size during optimization
    """
    currentData = data
    if isPCAEnabled :
        currentData = Init_PCA(data,True)[0]
    # Initialize t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, random_state=42)

    # Fit and transform data
    X_tsne = tsne.fit_transform(currentData)

    # Plotting t-SNE results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='Spectral', alpha=0.7)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization - Digits Dataset')
    plt.colorbar(scatter, label='Digit Label')
    plt.show()

    return X_tsne, "t-SNE"

## Select the right Dimension reduction (change if needed)
X_ReductedDimension , DimensionReductor = Init_UMAP()
# X2_ReductedDimension , D2imensionReductor = Init_tSNE(x,y,2,50,200)

## Print the reducted dimension shape
print(f"Original shape: {x.shape}")
print(f"{DimensionReductor}-reduced shape: {X_ReductedDimension.shape}") 

## FUNCTION TO SHOWS THE DATASET
def Plot_Dataset(num:int):    
    fig, axes = plt.subplots(1, num, figsize=((num+1), 3))
    for i, ax in enumerate(axes):
        ax.imshow(digits.images[i], cmap='viridis')
        ax.set_title(f"Label: {digits.target[i]}")
        ax.axis('off')

    plt.show()


## FUNCTION FOR CLUSTERING ALGORIGTHMS
def Init_Kmeans():
    ## Import K-Means
    from sklearn.cluster import KMeans
    global X_ReductedDimension

    ## initialize K-means to 10 clusters(0-9)
    kmeans =KMeans(n_clusters=10,random_state=42)

    ## fit kmeand to pca reduced data
    kmeans_labels = kmeans.fit_predict(X_ReductedDimension)

    ## Plot the K-Means Clustering result
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_ReductedDimension[:, 0], X_ReductedDimension[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.7)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('K-Means Clustering on PCA-Reduced Digits Data')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

    ## Calculate Silhouette Score
    score = silhouette_score(X_ReductedDimension, kmeans_labels)
    print(f"Silhouette Score for K-Means Clustering: {score:.2f}")

def Init_DBSCAN():
    ## Import K-Means
    from sklearn.cluster import DBSCAN

    # Initialize DBSCAN
    dbscan = DBSCAN(eps=0.1, min_samples=5)

    # Fit DBSCAN to PCA-reduced data
    dbscan_labels = dbscan.fit_predict(X_ReductedDimension)

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
        xy = X_ReductedDimension[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title(f'DBSCAN Clustering (Estimated Clusters: {n_clusters})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

    # Calculate Silhouette Score (ignoring noise points labeled as -1)
    if n_clusters > 1:
        score = silhouette_score(X_pca[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
        print(f"Silhouette Score for DBSCAN: {score:.2f}")
    else:
        print("Not enough clusters to compute Silhouette Score.")

def Init_HierarchicalClustering():
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram, linkage

    # Apply Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=10, linkage='ward')
    agg_labels = agg_clustering.fit_predict(X_ReductedDimension)

    # Calculate Silhouette Score
    score = silhouette_score(X_ReductedDimension, agg_labels)
    print(f"Silhouette Score for Hierarchical Clustering: {score:.2f}")

    # Plot Hierarchical Clustering results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_ReductedDimension[:, 0], X_ReductedDimension[:, 1], c=agg_labels, cmap='Spectral', s=30)
    plt.title("Hierarchical Clustering on PCA-Reduced Data")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label='Cluster')
    plt.show()

    # Generate linkage matrix
    linked = linkage(X_ReductedDimension, method='ward')

    # Plot Dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(linked, truncate_mode='level', p=5)  # Show only top 5 levels
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()

# Plot_Dataset(10)
# Init_Kmeans()
Init_DBSCAN()
# Init_HierarchicalClustering()