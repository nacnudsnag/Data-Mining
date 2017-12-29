import numpy as np
import urllib.request
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree                               
from sklearn.datasets import fetch_mldata         
import urllib.request
from sklearn.model_selection import KFold

def meanCalc(sample):
        #calculates means from an array of arrays
        means = []
        for each in sample:
                means.append(each.mean())
        return means


def confMatrix(model, X, y, dataset):
	"""Creates a confMatrix when given a model and the X and y"""
	#Creates the Test and Train split
        # use the model on the training data
        # (whatever your model variable was last set to)


	# random, if you prefer
	#k_fold = KFold(n_splits=5, shuffle=True, random_state=None)
	
	# pseudorandom (repeatable)
	k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
	
	misclassed = []
	for k, (train, test) in enumerate(k_fold.split(X, y)):
		#fit the model
		model.fit(X[train], y[train])
	
		#get the predictions for that model and that data
		y_model = model.predict(X[test])
		
		#get the accuracy of that model on that data
		score = accuracy_score(y[test], y_model)
		print("accuracy:", score)

		# new: figure out which examples are misclassified
		misclassified = np.where(y[test] != y_model)
		
		print("misclassified", misclassified)  # so that we know what that looks like
		
		# helpful: what does the test data look like? these are the indices
		print("test data indices", test)
		
		# maybe helpful: grab just those examples and check the shape
		testdata = dataset[test,:]
		print(testdata.shape)

		# here, make a confusion matrix of this training-testing split
		mat = confusion_matrix(y[test], y_model)
		print("Matrix is: ")
		print(mat)  # print the confusion matrix 

		
		# possibly while debugging, you might want to break after one interation

	for each in misclassified:
		misclassed.append(dataset[each])
	for each in misclassed:
		print(each)
	
def run_dt(maxDepth, X, y, dataset):
	"""Runs a decision tree on X and y with various MaxDepths"""
	variations = []
	labels = []
	for i in range(0, len(maxDepth)):
		# Create Decision tree (for each max depth)
		dt = tree.DecisionTreeClassifier(criterion='entropy',
						max_depth=maxDepth[i])
		model = dt.fit(X, y)
		#Add necessary label.
		labels.append(str(maxDepth[i]))
		#Print just the first confusion Matrix
		if i == 0:
			print("\nConfusion Matrix for Max Depth: ", maxDepth[i])
			confMatrix(model, X, y, dataset)
		variations.append(cross_val_score(dt, X, y, cv=5))


	
def main():
	# URL for the Pima Indians Diabetes dataset
	# random, if you prefer
	url = "http://archive.ics.uci.edu/ml/machine-learning-databases/"
	url += "pima-indians-diabetes/pima-indians-diabetes.data"
	
	# download the file
	raw_data = urllib.request.urlopen(url)
	
	# load the CSV file as a numpy matrix        
	dataset = np.loadtxt(raw_data, delimiter=",")
	#print(dataset.shape)   # if you like
	
	# separate the data from the target attributes                             
	X = dataset[:,0:7]
	y = dataset[:,8]
	run_dt([1000], X, y, dataset)





main()
