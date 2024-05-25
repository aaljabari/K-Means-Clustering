from utils import plot_costs, plot_silhouette_scores, plot_davies_bouldin_scores
from utils import load_dataset
from kmeans import KMeans  # Corrected import statement

datasets_info = [
    ("iris", 3),
    ("3gaussians-std0.6", 3),
    ("3gaussians-std0.9", 3),
    ("circles", 2),
    ("moons", 2)
]

datasets = {}
results = {}

# Process each dataset
for dataset_name, k in datasets_info:
    X, y = load_dataset(dataset_name)
    datasets[dataset_name] = (X, y)
    
    kmeans = KMeans(k=k)
    centroids, nearest_centroids, _ = kmeans.fit(X)
    
    results[(dataset_name, k)] = {
        'centroids': centroids,
        'nearest_centroids': nearest_centroids,
        'cost': kmeans.losses,
        'silhouette': kmeans.silhouettes,
        'davies_bouldin': kmeans.davies_bouldins
    }

# Plot the evaluation metrics
plot_costs(results)
plot_silhouette_scores(datasets, results)
plot_davies_bouldin_scores(datasets, results)
