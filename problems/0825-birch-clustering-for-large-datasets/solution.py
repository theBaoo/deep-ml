import numpy as np

def birch_cluster(X, threshold):
    """
    Single-level BIRCH clustering.
    X: array-like of shape (n_samples, n_features)
    threshold: float, max allowed subcluster radius
    Returns: list of centroids (each a list of floats), sorted lexicographically.
    """

    subclusters = []

    for x in X:
        x = np.asarray(x, dtype=float)
        if not subclusters:
            subclusters.append(new_cluster_from(x))
            continue
        
        idx = nearest(subclusters, x)
        cluster = subclusters[idx]

        new_N = cluster["N"] + 1
        new_LS = cluster["LS"] + x
        new_SS = cluster["SS"] + x * x

        variance = (
            new_SS / new_N
            - (new_LS / new_N) ** 2
        )
        variance = np.maximum(variance, 0)
        radius = np.sqrt(np.sum(variance))

        if radius <= threshold:
            cluster["N"] = new_N
            cluster["LS"] = new_LS
            cluster["SS"] = new_SS
        else:
            subclusters.append(new_cluster_from(x))
    
    centroids = [
        cluster["LS"] / cluster["N"]
        for cluster in subclusters
    ]
    # centroids.sort()

    return [c.tolist() for c in centroids]
    

def new_cluster_from(x):
    return {
        "N": 1,
        "LS": x,
        "SS": x * x
    }

def nearest(s, x):
    distances = []
    for cluster in s:
        centroid = cluster["LS"] / cluster["N"]
        distance = np.linalg.norm(x - centroid)
        distances.append(distance)

    return np.argmin(distances)


