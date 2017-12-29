"""
Lab 9, CSCI 2600
Starter code from Clare

Just to get the digits data in there
"""

##### for Lab9, try the digits first
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
import matplotlib.pyplot as plt

def neuralNet(hiddenNodes):
	#function to create neural net with given hidden Nodes
	model = MLPClassifier(hidden_layer_sizes=hiddenNodes, # tuple;default (100,)
                              activation='logistic',      # default is 'relu'
                              max_iter = 500,             # default is 200
                              random_state=0)             # seed
	return model
def predAccuracy(model, scaler, X, y):
	#returns cross val scores for neural net, option to use scaler
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                        test_size=.2,    # 20%
                                                        random_state=0)  # seed

	if scaler:
		# for NN's, need to have all values on [0..1] range; 
	        #   StandardScaler does this for us 
	        scaler = StandardScaler()
	        # Fit only to the training data
	        scaler.fit(Xtrain)
	
	        # Now apply the transformations to the data:
	        Xtrain = scaler.transform(Xtrain)
	        Xtest = scaler.transform(Xtest)
	
	        # will need to scale all of X, for cross-val
	        X = scaler.transform(X)
	        # Train the classifier with the training data
	        model.fit(Xtrain,ytrain)
	
	        # predictions 
	        y_model = model.predict(Xtest)
	
        	# get the prediction accuracy for this run
        	score = cross_val_score(model, X, y, cv=5)
        	return score
	else:
		model.fit(Xtrain,ytrain)
        	# predictions 
		y_model = model.predict(Xtest)

        	# get the prediction accuracy for this run
		score = cross_val_score(model, X, y, cv=5)
		return score

def meanCalc(sample):
	#calculates means from an array of arrays
        means = []
        for each in sample:
                means.append(each.mean())
        return means


def confIntervals(samples):
	#returns confidence intervals from an array of arrays
        z_critical = stats.norm.ppf(q = 0.975)
        means = meanCalc(samples)
        error = []
        for sample in samples:
                error.append(z_critical * sample.std() / np.sqrt(len(samples)))
        confInterv = []
        i = 0
        while i < len(samples):
                confInterv.append([means[i] - error[0], means[i] + error[0]])
                i+=1
        return confInterv
def printPs(sample, sample2, variations):
	#Prints P value when given two sets of data
        pair = stats.ttest_rel(variations[sample], variations[sample2])
        print("The p-value for {0}, {1} is {2}".format(sample, sample2, pair[1]))


def main():
	#Add Data
	digits = load_digits()
	print("Digits shape is: ", digits.images.shape)
	
	X = digits.data
	print("X shape is", X.shape)
	
	y = digits.target
	print("y shape is", y.shape)
	
	
	############################################################
	## MLP: multi-layer perceptron
	
	# Initialize ANN classifier
	# Note: all of these parameters have default values
	#   and should therefore be optional
	model = neuralNet((10, 10))	
	#check difference with and without scaler
	print("Without Scaler")
	print(predAccuracy(model, False, X, y))
	print("With Scaler")
	print(predAccuracy(model, True, X, y))
	variations = []
	# Check all the variations of hidden Nodes 
	for i in range(1, 6):
		print("net with three layers of ", i * 10)
		model = neuralNet((10 * i, 10 * i, 10 * i))
		accuracy = predAccuracy(model, True, X, y)
		print(accuracy)
		variations.append(accuracy)
	#Calculate Intervals
	sample_means = meanCalc(variations)
	intervals = confIntervals(variations)

	plt.figure(figsize=(9,9))

	xvals = np.arange(5, 30, 5)
	yerrors = [(top-bot)/2 for top,bot in intervals]
	fig = plt.figure()
	plt.errorbar(x=xvals,
	             y=sample_means,
	             yerr=yerrors,
	             fmt='D')
	labels = ["h=10", "h=20", "h=30", "h=40", "h=50"]
	plt.xticks(xvals,labels)
	# use this to write to a file; look at the file with display  
	fig.savefig("netVariations.png")
	for i in range(0, 5):
	        for j in range(i + 1, 5):
	                printPs(i, j, variations)
main()
