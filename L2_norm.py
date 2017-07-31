import numpy as np
import scipy.spatial

def getDistance(a, b):
    return np.linalg.norm(a-b)

def getAllDistance(data, centroids):
    return scipy.spatial.distance.cdist(centroids, data, 'euclidean')

def getClasterDistance(data, clusters, centroids):

    dists = 0.0

    for centroid_num in range(centroids.shape[0]):
        for clusters_num in range(clusters.shape[0]):
            if clusters[clusters_num] == centroid_num:
                dists += getDistance(centroids[centroid_num], data[clusters_num])

    return dists