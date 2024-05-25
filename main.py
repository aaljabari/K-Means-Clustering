import numpy as np
from utils import plot_pca_iris_centroids, load_dataset, plot_kmeans_clusters,plot_evaluation_metrics, plot_original_and_predicted_clusters
from kmeans import KMeans  # Corrected import statement

# 'iris'
X, y = load_dataset('3gaussians-std0.9')#("3gaussians-std0.6")

kmeans = KMeans(k=2)
centriods,nearest_centroids,error = kmeans.fit(X)

#plot_pca_iris_centroids(X, y, centriods)
#plot_kmeans_clusters(X, y, centriods,nearest_centroids)
plot_original_and_predicted_clusters(X, y, nearest_centroids,centriods)
plot_evaluation_metrics(kmeans.losses, kmeans.silhouettes, kmeans.davies_bouldins)


