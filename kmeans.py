import numpy as np
from scipy.spatial.distance import pdist
import itertools
from sklearn.metrics import silhouette_score, davies_bouldin_score

class KMeans:
    def __init__(self, k):
        self.k = k
        self.centroids = None
        self.j = float('inf')
        self.losses = []
        self.silhouettes = []
        self.davies_bouldins = []

    def initialize_selected_clusters(self, data): # select random points 
        n_samples, _ = data.shape
        centroids_indices = np.random.choice(n_samples, self.k, replace=False) 
        self.centroids = np.asarray(data[centroids_indices])

    def initialize_clusters(self, data): # random initialization
        _, n_features = data.shape
        self.centroids = np.random.rand(self.k, n_features) * (data.max() - data.min()) + data.min()

    def initialize_distant_centroids(self, data): # distant points
        c = [list(x) for x in itertools.combinations(range(len(data)), self.k)]
        distances = []
        for i in c:    
            distances.append(np.mean(pdist(data[i, :])))
        
        ind = distances.index(max(distances))
        rows = c[ind]
        self.centroids = data[rows]

    def find_nearest_centroid(self, data):
        nearest_centroids = np.zeros(len(data), dtype=int)
        
        for i, point in enumerate(data):
            min_distance = float('inf')
            nearest_centroid_idx = -1
            
            for j, centroid in enumerate(self.centroids):
                distance = np.sqrt(np.sum((point - centroid) ** 2))
                if distance < min_distance:
                    min_distance = distance
                    nearest_centroid_idx = j
            
            nearest_centroids[i] = nearest_centroid_idx
        
        return nearest_centroids
    
    def assign_labels(self, X):
        print('------------')
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, data, nearest_centroids):
        centroids = np.zeros((self.k, data.shape[1]))
        for cluster_idx in range(self.k):
            cluster_points = data[nearest_centroids == cluster_idx]
            if len(cluster_points) > 0:
                centroids[cluster_idx] = np.mean(cluster_points, axis=0)
            else:
                centroids[cluster_idx] = self.centroids[cluster_idx]
        self.centroids = centroids  

    def objective_function(self, data, nearest_centroids):
        total_distance = 0
        for cluster_idx, centroid in enumerate(self.centroids):
            cluster_points = data[nearest_centroids == cluster_idx]
            total_distance += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
        self.j = total_distance
        self.losses.append(total_distance)

    def evaluation(self, data, nearest_centroids):
        silhouette = silhouette_score(data, nearest_centroids) # higher is better (ranging from -1 to 1)
        davies_bouldin = davies_bouldin_score(data, nearest_centroids) # lower is better
        self.silhouettes.append(silhouette)
        self.davies_bouldins.append(davies_bouldin)
        print(f"    -------> Cost: {self.j}")
        print(f"    -------> silhouette: {silhouette}")
        print(f"    -------> davies_bouldin: {davies_bouldin}\n\n")    

    def fit(self, data):
        self.initialize_distant_centroids(data)
        #nearest_centroids = self.find_nearest_centroid(data)
        nearest_centroids=self.assign_labels(data)

        self.objective_function(data, nearest_centroids)
        self.evaluation(data, nearest_centroids)
        epsilon = self.j
        i = 1
        while epsilon > .0001:
            print(f"Iteration no.: {i}")
            i += 1
            #nearest_centroids = self.find_nearest_centroid(data)
            nearest_centroids=self.assign_labels(data)
            self.update_centroids(data, nearest_centroids)
            self.objective_function(data, nearest_centroids)
            self.evaluation(data, nearest_centroids)
            epsilon -= self.j
            
        return self.centroids, nearest_centroids, self.j
