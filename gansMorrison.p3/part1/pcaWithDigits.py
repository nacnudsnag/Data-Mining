import matplotlib.pyplot as plt      # we'll need this to plot
from sklearn.datasets import load_digits
from sklearn import decomposition
from sklearn.cluster import KMeans
"""
In the first Principle Component Analysis visualization, the file is simply a 
way to see complex multidimensional data in only two dimensions, measured by
component one and two. Intially there is still clustering, but the clustering 
isn't based on the k-means algorithm. Therefore, there is still some aspect of
clustering, but lots of overlap between clusters that are not very defined.
The plot from the K-Means test is the result of running K-Means clustering 
on the Principle Components of the data. The algorithm doesn't use the actual
data from digits, but rather the converted principle components. Because of
this, the actual clusters are different in each of the two figures.
"""
def main():
	digits = load_digits()

	X = digits.data
	print("X shape is", X.shape)
	y = digits.target
	print("y shape is", y.shape)

	pca = decomposition.PCA(n_components=2) # project from 64 to 2 dimensions
	
	# get the transformed X and print the shape
	Xprojected = pca.fit_transform(X)
	print("Xprojected.shape is", Xprojected.shape)

	fig = plt.figure()

	# plot it
	#   here, we are plotting the attribute 0 on the X axis, 
	#     and attribute 1 on the y axis
	#   The coloring follows the actual classes, y (c=y)
	plt.scatter(Xprojected[:, 0], Xprojected[:, 1],    # x and y axis of plot
	            c=y,                                   # set the color
	            cmap='viridis')                        # color scheme
	plt.xlabel('component 1')
	plt.ylabel('component 2')
	
	fig.savefig("PCA.png")

	kmeans = KMeans(n_clusters=10,     # 10 clusters
	                n_init=1,           # but only repeat one time (one search)
	                init='random'       # we'll start at random, not "smart"
	               )
	kmeans.fit(X)
	y_kmeans = kmeans.predict(X)
	
	kmeans.fit(Xprojected)                  # fit the projected data (2d)
	y_kmeans = kmeans.predict(Xprojected)   # get the cluster ID
	
	# plot the points, with colors
	plt.scatter(Xprojected[:, 0], Xprojected[:, 1], 
	            c=y_kmeans,                 # cluster ID, not actual class
	            cmap='viridis')
	plt.xlabel('component 1')
	plt.ylabel('component 2')
	
	fig.savefig("K-Means.png")

main()


