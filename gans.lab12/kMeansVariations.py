import matplotlib.pyplot as plt
import seaborn as sns; sns.set()       # for plot styling
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);
        

def kmeans(clusters, initNum, dumb, state, maxIter):
        if dumb:
                kmeans = KMeans(n_clusters=clusters,
                                n_init = initNum,
                                init='random',
                                random_state = state,
                                max_iter = maxIter)
        else:
                kmeans = KMeans(n_clusters=clusters,
                                n_init = initNum,
				max_iter = maxIter)

        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)

        # plot the points, with colors
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
        # plot the centers
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
        plt.show()

def main():
        #kmeans(4, 1, True, 0, 300) #Smart Centroids

	"""Here the function works perfectly, the centroids are right in the
	middle of the most obvious clusters"""

        #kmeans(4, 1, True, 0, 300) #Dumb
	"""When making the k-means function "dumb" it places centroids in an 
	innoportune place, and with only 1 repetition doesn't let it change
	much"""

        #kmeans(4, 100, True, 0, 300) #Dumb but with enough repetitions to fix
	""" Here, even though the inital centroid placing was bad, with so many
	repititions and a high Max Iter value it brought the clusters back to 
	how they were when the centroids where chosen not randomly"""

	#kmeans(10, 1, True, 00, 300) #10 Clusters
	"""With ten different clusters, the fundamental goals of clustering 
	aren't really reached. with 10, you cannot really maximize differences
	between classes. Although the differences within classes are small, 
	the clustering isn't that effective"""
	
	"""Generalizations:

	Max Iter Value: Allows the clusters to shape towards the most efficient
	manner by increasing this value. If set too low, badly placed centroids
	don't have enough iterations to fix themeselves.

	Smart v. Dumb: Decides where the initial centroids are placed, by 
	setting it to random, the clustering is less effective unless you 
	allow it a lot of repititions and iterations. 

	n_init: The amount of repetitions, increasing this, allows badly placed
	centroids (When Max iter is also high) to fix themeselves and revert to
	the best placing

	Number of Clusters: Changes the amount of clusters, there is a sweet 
	spot for the most reasonable number of clusters based on the data.
	10 is way too many clusters for the given data set."""


main()
