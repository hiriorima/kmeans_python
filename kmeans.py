import numpy as np
import L2_norm as euclid_distance

def cluster_centroids(data, clusters, k):

    result = np.empty(shape=(k,) + data.shape[1:])
    for i in range(k):
        np.mean(data[clusters == i], axis=0, out=result[i])
    return result

def kmeans(data, k=None, steps=100):

    prev_dist = 10000
    result = None

    for _ in range(1000):

        centroids = data[np.random.choice(np.arange(len(data)), k, False)]

        for _ in range(max(steps, 1)):
            dists = euclid_distance.getAllDistance(data, centroids)
            clusters = np.argmin(dists, axis=0)

            new_centroids = cluster_centroids(data, clusters, k)
            if np.array_equal(new_centroids, centroids):
                break

            centroids = new_centroids
            
            current_distance = euclid_distance.getClasterDistance(data, clusters, centroids)
            if current_distance < prev_dist:
                prev_dist = current_distance
                result = clusters
    return result