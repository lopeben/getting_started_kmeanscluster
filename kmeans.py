import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


## Synthetic data
features, true_labels = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=2.75,
    random_state=42
)

## In this example, you’ll use the StandardScaler class. 
## This class implements a type of feature scaling called standardization. 
## Standardization scales, or shifts, the values for each numerical feature in your dataset 
## so that the features have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(scaled_features)