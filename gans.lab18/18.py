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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint


# Utility function to report best scores
#   default: top three ranks... but maybe more than three, because of ties
def report(results, n_top=3):
    for i in range(1, n_top + 1):

        # this line makes a list of the items of this rank
        candidates = np.flatnonzero(results['rank_test_score'] == i)

        print("candidates are:", candidates)
        print("rank_test_score is:", results['rank_test_score'])
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def randoParaSearch(skf, model, parameters, X, y):
        # run randomized search
	n_iter_search = 2         # number of random combinations to look at
	random_search = RandomizedSearchCV(model,                             #RF model
	                                   param_distributions=parameters, #params
	                                   n_iter=n_iter_search,           #samples
	                                   cv = skf)                       #CV model


	# start a timer, run through all the combinations, print some info
	start = time()
	random_search.fit(X, y)
	print("RandomizedSearchCV took %.2f seconds for %d candidates"
	      " parameter settings." % ((time() - start), n_iter_search))
	report(random_search.cv_results_)

def gridParaSearch(skf, model, parameters, X, y):
	# run grid search          
	grid_search = GridSearchCV(model,                      # RF model
	                           param_grid=parameters,   # parameters to vary
	                           cv = skf)                # CV model

	# start a timer, run through all the combinations, print some info
	start = time()
	grid_search.fit(X, y)
	print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
	      % (time() - start, len(grid_search.cv_results_['params'])))


	report(grid_search.cv_results_)


def main():

	digits = load_digits()

	print("Digits shape is: ", digits.images.shape)

	X = digits.data
	print("X shape is", X.shape)

	y = digits.target
	print("y shape is", y.shape)

	#define with the parameters that will be fixed
	model = RandomForestClassifier(criterion='entropy',
		                    random_state=0)

	# the parameters that will be varied  
	parameters = {'n_estimators': [10, 50],
        		'max_depth':[5,10]
			}


	# create a cross-validator, for repeatable splits
	skf = StratifiedKFold(n_splits=10,        # 10 splits
	                      shuffle=True,       # shuffle the data
	                      random_state=0)     # fix the seed, so it's repeatable
	
	print("GRID SEARCH")
	gridParaSearch(skf, model, parameters, X, y)
	print("RANDOM SEARCH")  
	randoParaSearch(skf, model, parameters, X, y)

	# Create ANN classifier
	ann = MLPClassifier(activation='logistic',      # default is 'relu'
        	            random_state=0)             # seed 

	# the parameters that will be varied
	parameters = {'hidden_layer_sizes': [(50),                   # default is (100,)
	                                     (100),
	                                     (200),
	                                     (10,10),
	                                     (50,50),
	                                     (100,100),
	                                     (200,200)],
	              'max_iter': [500,1000,2000,4000]     # default is 200 
	              }

	print("GRID SEARCH")
	gridParaSearch(skf, ann, parameters, X, y)
	print("RANDOM SEARCH")
	randoParaSearch(skf, ann, parameters, X, y)


main()
