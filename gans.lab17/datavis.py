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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.datasets import fetch_mldata
import urllib.request

def run_rf(seeds, X, y):
	"""Runs a Random Forest on X and y with various seeds"""
	# Create Random Forest (for each seed)
	rf = RandomForestClassifier(n_estimators = 100,
	                                        random_state=seeds,
	                                        criterion='entropy')
	model = rf.fit(X, y)
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
	
	scores = cross_val_score(rf, X, y, cv=5)
	print("SCORESSSS")
	print(scores)
	
	# plot a dot for each c-v run and a line for the mean across the cv runs
	import matplotlib.pyplot as plt
	# plot a dot for each c-v run and a line for the mean across the cv runs
	plt.scatter(np.arange(len(scores)), scores, label='accuracy')
	plt.axhline(y=np.mean(scores), color='g', label='mean')
	
	# make title, axes labels, and legend
	plt.title('Cross validation scores')
	plt.xlabel('Cross validation run number')
	plt.ylabel('Accuracy')
	plt.legend()

	# plot it
	plt.show()
	
	# learning curves
	from sklearn.model_selection import learning_curve
	
	spacing = np.linspace(0.1, 1.0, 10)  # .1 to 1.0; 10 values
	train_sizes, train_scores, test_scores = learning_curve(estimator=rf,
	                                                        X=X,
	                                                        y=y,
	                                                        train_sizes=spacing,
	                                                        cv=10)
	
	#note: train_scores and test_scores are 2d arrays
	# axis=1 below gets the rows (vs. the columns)
	
	# mean and standard deviation of accuracy on training data
	train_mean = np.mean(train_scores, axis=1)
	train_std = np.std(train_scores, axis=1)
	
	# mean and standard deviation of accuracy on testing data
	test_mean = np.mean(test_scores, axis=1)
	test_std = np.std(test_scores, axis=1)
	# plot training accuracies
	plt.plot(train_sizes,                         # x
	         train_mean,                          # y
	         color='red',                         # red
	         marker='o',                          # with dots
	         label='training accuracy')           # for legend
	
	# plot the variance of training accuracies -- red shading
	plt.fill_between(train_sizes,                 # x
	                 train_mean+train_std,        # ymax
	                 train_mean-train_std,        # ymin
	                 alpha=0.15,                  # shading
	                 color='red')
	# plot training accuracies
	plt.plot(train_sizes,                         # x
	         test_mean,                          # y
	         color='blue',                         # red
	         marker='o',                          # with dots
	         label='test accuracy')           # for legend
	
	# plot the variance of training accuracies -- red shading
	plt.fill_between(train_sizes,                 # x
	                 test_mean+test_std,        # ymax
	                 test_mean-test_std,        # ymin
	                 alpha=0.15,                  # shading
	                 color='blue')


	
	
	# ------------------------------------------------------------
	# validation curve
	from sklearn.model_selection import validation_curve
	
	# parameter range -- run the model repeatedly with these different values
	p_range = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
	train_scores, test_scores = validation_curve(estimator=rf,
	                                             X=X,
	                                             y=y,
	                                             param_name='n_estimators',
	                                             param_range=p_range,
	                                             cv=10)
	
	# Mean value of accuracy against training data
	train_mean = np.mean(train_scores, axis=1)
	
	# Standard deviation of training accuracy per number of training samples
	train_std = np.std(train_scores, axis=1)
	
	# Same as above for test data
	test_mean = np.mean(test_scores, axis=1)
	test_std = np.std(test_scores, axis=1)
	
	
		
	
	plt.show()

def countInstances(y):
	vals = set(y)
	count = []
	for i in range(0, len(vals)):
		count.append(0)
	allVals = list(y)
	for each in allVals:
	        count[each] += 1
	print(count)

def main():
	digits = load_digits()
	print("Digits shape is: ", digits.images.shape)

	X = digits.data
	print("X shape is", X.shape)
	
	y = digits.target
	print("y shape is", y.shape)
	countInstances(y)
	run_rf(1, X, y)

main()	
