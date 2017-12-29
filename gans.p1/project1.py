"""
Name: Duncan Gans
Date: March 10, 2017

This code contains all the functions needed to run and measure several
different classification approaches. The functions are split into two groups.
The first set is functions that are used to show and display the effects of
using different classifications. This includes functions related to showing
confidence intervals, confusion matrices, and calculating P values. The other
group of functions are the actual functions that create models of Random
forests, decision trees, etc. and then run the functions described above that
show the results. Combined, these functions let you run various classification
approaches with varying inputs to compare them, while returning meaningful 
results. To use the code, simply run it using python3. This will print out the
results of each of the classification approaces on wine data, and create
 confidence intervals externally.
"""
#import necessary libraries, additions
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

def meanCalc(sample):
        #calculates means from an array of arrays
        means = []
        for each in sample:
                means.append(each.mean())
        return means


def confIntervals(samples):
        #returns confidence intervals from an array of arrays
        z_critical = stats.norm.ppf(q = 0.975)
        means = meanCalc(samples) #uses means from meanCalc function
        error = []
        for sample in samples:
		#calculates the error first
                error.append(z_critical * sample.std() / np.sqrt(len(samples)))
        confInterv = []
        i = 0
        while i < len(samples):
		#adds and subtracts the error from the mean to get intervals
                confInterv.append([means[i] - error[0], means[i] + error[0]])
                i+=1
        return confInterv
def printPs(sample, sample2, variations):
	#Prints P value when given two ints, and a set of data,
	#returns the p values for the two indices given
	pair = stats.ttest_rel(variations[sample], variations[sample2])
	if not np.isnan(pair[1]):
		print(pair[1])
	else:
		#if P value is nan (i.e. samples are identical) print message.
		print("Scores are Identical")
def showConfInterv(fileName, variations, labels):
	#When given a set of five-fold cross evaluation scores it calculates
	#confidence intervals and creates a file with the given filename
	#and labels
	sample_means = meanCalc(variations)
	intervals = confIntervals(variations) #calculate intervals and means

	#begin creating graph
	plt.figure(figsize=(9,9))

	xvals = np.arange(5, 30, 5)
	yerrors = [(top-bot)/2 for top,bot in intervals]
	fig = plt.figure()
	plt.errorbar(x=xvals,
	             y=sample_means,
	             yerr=yerrors,
	             fmt='D')
	#add labels
	plt.xticks(xvals,labels)
	# use this to write to a file; look at the file with display  
	fig.savefig(fileName)
def confMatrix(model, X, y):
	"""Creates a confMatrix when given a model and the X and y"""
	#Creates the Test and Train split
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
	                                        test_size=.2,
	                                        random_state=10)
        # use the model on the training data
        # (whatever your model variable was last set to)
	model.fit(Xtrain, ytrain)
	y_model = model.predict(Xtest)

	#Get accuracy score and print it out
	score = accuracy_score(ytest, y_model)
	print("Score is: ")
	print(score)

        ### confusion matrix:
	mat = confusion_matrix(ytest, y_model,
				 labels=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
					6.0, 7.0, 8.0, 9.0, 10.0])
	print("Matrix is: ")
	print(mat)  # print the confusion matrix 
		
def run_dt(maxDepth, X, y):
	"""Runs a decision tree on X and y with various MaxDepths"""
	variations = []
	labels = []
	confMatrixH()
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
			confMatrix(model, X, y)
		variations.append(cross_val_score(dt, X, y, cv=5))
	showConfInterv("decisionTree.png", variations, labels)
	pHeader()
	allPs(variations)

def run_rf(seeds, X, y):
	"""Runs a Random Forest on X and y with various seeds"""
	variations = []
	confMatrixH() #Header
	labels = []
	for i in range(0, len(seeds)):
		# Create Random Forest (for each seed)
		rf = RandomForestClassifier(n_estimators = 100,
						random_state=seeds[i],
						criterion='entropy')
		model = rf.fit(X, y)
		#Add necessary label
		labels.append(str(seeds[i]))
		#print just the first confusion Matrix
		if i == 0:
	        	print("\nConfusion Matrix for a seed of ", seeds[i])
	        	confMatrix(model, X, y)
		variations.append(cross_val_score(rf, X, y, cv=5))
	showConfInterv("randomForests.png", variations, labels)
	pHeader() #Header
	allPs(variations)

def run_neighbors(ks, X, y):
	"""Runs a K nearest Neighbors on X and y for various k values"""
	variations = []
	confMatrixH() #Header
	labels = []
	for i in range(0, len(ks)):
		#Create K Nearest Neighbors (For each K value)
		model = KNeighborsClassifier(n_neighbors=ks[i],
						weights='uniform')
		if i == 0:
			#Print the first confusion Matrix
			print("\nConfusion Matrix for a k of : ", ks[i])
			confMatrix(model, X, y)
		score = cross_val_score(model, X, y, cv=5)
		#Add necessary label
		labels.append(str(ks[i]))
		variations.append(score)
	showConfInterv("NearestNeighbors.png", variations, labels)
	pHeader() #Header
	allPs(variations)

def run_ANN(iterations, layers, X, y):
	"""Runs a Neural Net on X and y for various layers and iterations"""
	variations = []
	confMatrixH() #Header
	labels = []
	for i in range(0, len(iterations)):
		#create neural net with given hidden layers and iterations
		model = MLPClassifier(hidden_layer_sizes = layers[i],
					activation = 'logistic',
					max_iter = iterations[i],
					random_state=0)
		if i == 0:
			#Print just the first Confusion Matrix
			score = cross_val_score(model, X, y, cv=5)
			print("\nConfusion Matrix for layers: ",layers[i], 
				"iterations: ", iterations[i])
			confMatrix(model, X, y)
		#Add necessary labels
		labels.append((layers[i][0], iterations[i]))
		variations.append(score)
	showConfInterv("NeuralNet.png", variations, labels)
	pHeader() #Header
	allPs(variations)
def pHeader():
	#Header for P Values
	print("                   P Values                 ")
	print("--------------------------------------------")
def confMatrixH():
	#Header for Confusion Matrix
	print("             Confusion Matrix               ")
	print("--------------------------------------------")
def allPs(variations):
	#Print all of the P values of a given set of cross val variations
	print("Pair 1 and 2")
	printPs(0, 1, variations)	
	print("Pair 1 and 3")
	printPs(0, 2, variations)
	print("Pair 1 and 4")
	printPs(0, 3, variations)
	print("Pair 1 and 5")
	printPs(0, 4, variations)
	print("Pair 2 and 3")
	printPs(1, 2, variations)
	print("Pair 2 and 4")
	printPs(1, 3, variations)
	print("Pair 2 and 5")
	printPs(1, 4, variations)
	print("Pair 3 and 4")
	printPs(2, 3, variations)
	print("Pair 3 and 5")
	printPs(2, 4, variations)
	print("Pair 4 and 5")
	printPs(3, 4, variations)

def main():
	#Get necessary data
	url = "http://www.bowdoin.edu/~congdon/Courses/2600/Data/Wine/winequality-white.csv"
	raw_data = urllib.request.urlopen(url)
	dataset = np.loadtxt(raw_data, delimiter=";", dtype='float64')
	X = dataset[:,0:10]
	y = dataset[:,11]
	print("X shape is", X.shape)
	print("y shape is", y.shape)
	print("--------------------------------------------")
	print("               Decision Trees               ")
	print("--------------------------------------------")

	run_dt([5, 6, 7, 8, 9], X, y)
	print("--------------------------------------------")
	print("               Random Forests               ")
	print("--------------------------------------------")
	run_rf([5, 6, 7, 8, 9], X, y)
	print("--------------------------------------------")
	print("            K-Nearest Neighbors             ")
	print("--------------------------------------------")
	run_neighbors([5, 7, 9, 11, 13], X, y)
	print("--------------------------------------------")
	print("               Neural Net                   ")
	print("--------------------------------------------")
	run_ANN([100, 200, 300, 400, 500],[[10, 10, 10], [20, 20, 20],
		 [30, 30, 30], [40, 40, 40], [50, 50, 50]], X, y)

main()






