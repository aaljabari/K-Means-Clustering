import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

import pandas as pd

def plot_2d_pca_iris(X, y):
    plt.figure(1, figsize=(8, 6))
    plt.clf()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    for name, label, color in [("Setosa", 0, 'red'), ("Versicolor", 1, 'blue'), ("Virginica", 2, 'green')]:
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=name, color=color)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Iris Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pca_iris_centroids(X, y, initial_centroids=None):
    plt.figure(1, figsize=(8, 6))
    plt.clf()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    for name, label, color in [("Setosa", 0, 'red'), ("Versicolor", 1, 'blue'), ("Virginica", 2, 'green')]:
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=name, color=color)
    
    if initial_centroids is not None:
        initial_centroids_pca = pca.transform(initial_centroids)
        plt.scatter(initial_centroids_pca[:, 0], initial_centroids_pca[:, 1], 
                    c='black', marker='x', s=200, linewidths=2, label='Initial Centroids')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Iris Dataset with Initial Centroids')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_3d_pca_iris(X, y):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])
    plt.cla()
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
        ax.text3D(
            X_pca[y == label, 0].mean(),
            X_pca[y == label, 1].mean() + 1.5,
            X_pca[y == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
        )
    
    y = np.choose(y, [1, 2, 0]).astype(float)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    plt.show()

def load_dataset(i):
    if i == "iris":
        iris = load_iris()
        X = iris.data
        y = iris.target

    else:
        if i == '3gaussians-std0.6':
            df = pd.read_csv("datasets/3gaussians-std0.6.csv")
        elif i == '3gaussians-std0.9':
            df = pd.read_csv("datasets/3gaussians-std0.9.csv")
        elif i == 'circles':
            df = pd.read_csv("datasets/circles.csv")
        elif i == 'moons':
            df = pd.read_csv("datasets/moons.csv")
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values 

        #scaler = StandardScaler()
        #X = scaler.fit_transform(X)

    return X, y

def plot_kmeans_clusters(X, y, centroids=None, cluster_assignments=None):
    plt.figure(1, figsize=(8, 6))
    plt.clf()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    '''
    # Plotting original data points with their original labels
    for name, label, color in [("Setosa", 0, 'red'), ("Versicolor", 1, 'blue'), ("Virginica", 2, 'green')]:
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=name, color=color)
    '''
    # Plotting centroids if provided
    if centroids is not None:
        centroids_pca = pca.transform(centroids)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                    c='black', marker='x', s=200, linewidths=2, label='Centroids')
    
    # Plotting data points with their assigned clusters
    if cluster_assignments is not None:
        unique_clusters = np.unique(cluster_assignments)
        cluster_colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_clusters)))
        for cluster, color in zip(unique_clusters, cluster_colors):
            plt.scatter(X_pca[cluster_assignments == cluster, 0], 
                        X_pca[cluster_assignments == cluster, 1], 
                        label=f'Cluster {cluster}', color=color)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Iris Dataset with Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_evaluation_metrics(losses, silhouettes, davies_bouldins):
    iterations = range(len(losses))
    
    plt.figure(figsize=(8, 6))  # Smaller figure size
    
    plt.subplot(3, 1, 1)
    plt.plot(iterations, losses, marker='o', linestyle='-', color='blue', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('K-means Cost')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(iterations, silhouettes, marker='o', linestyle='-', color='green', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Silhouette Score')
    plt.title('K-means Silhouette Score')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(iterations, davies_bouldins, marker='o', linestyle='-', color='red', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('K-means Davies-Bouldin Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_original_and_predicted_clusters1(X, y_original, y_pred):
    # Apply PCA to reduce the dataset to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original clusters
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_original, cmap='viridis', edgecolor='k')
    axes[0].set_title('Original Clusters')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].grid(True)
    
    # Plot predicted clusters
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', edgecolor='k')
    axes[1].set_title('Predicted Clusters')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()



def plot_original_and_predicted_clusters(X, y_original, y_pred, centroids):
    # Apply PCA to reduce the dataset to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centroids_pca = pca.transform(centroids)
    
    # Define colors
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original clusters
    for label in np.unique(y_original):
            axes[0].scatter(X_pca[y_original == label, 0], X_pca[y_original == label, 1], 
                            color=colors[int(label) % len(colors)], label=f'Cluster {int(label)}', edgecolor='k')

    axes[0].set_title('Original Clusters')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot predicted clusters and centroids
    for label in np.unique(y_pred):
        axes[1].scatter(X_pca[y_pred == label, 0], X_pca[y_pred == label, 1], 
                        color=colors[int(label) % len(colors)], label=f'Cluster {int(label)}', edgecolor='k')
    
    # Plot centroids
    axes[1].scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                    color='black', marker='x', s=200, label='Centroids')
    axes[1].set_title('Predicted Clusters')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


#========================
from sklearn.metrics import silhouette_score, davies_bouldin_score

def plot_costs(results):
    plt.figure(figsize=(10, 6))
    for (dataset_name, k), data in results.items():
        costs = data['cost']
        plt.plot(range(len(costs)), costs, label=f'{dataset_name} (k={k})')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('K-means Cost for Different Datasets')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_silhouette_scores(datasets, results):
    plt.figure(figsize=(10, 6))
    for (dataset_name, k), data in results.items():
        X = datasets[dataset_name][0]
        nearest_centroids = data['nearest_centroids']
        silhouettes = data['silhouette']
        plt.plot(range(len(silhouettes)), silhouettes, label=f'{dataset_name} (k={k})')
    plt.xlabel('Iteration')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Datasets')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_davies_bouldin_scores(datasets, results):
    plt.figure(figsize=(10, 6))
    for (dataset_name, k), data in results.items():
        X = datasets[dataset_name][0]
        nearest_centroids = data['nearest_centroids']
        davies_bouldins = data['davies_bouldin']
        plt.plot(range(len(davies_bouldins)), davies_bouldins, label=f'{dataset_name} (k={k})')
    plt.xlabel('Iteration')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Score for Different Datasets')
    plt.legend()
    plt.grid(True)
    plt.show()