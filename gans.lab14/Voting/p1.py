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

        ### confusion matrix:
        mat = confusion_matrix(ytest, y_model,
                                 labels=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                        6.0, 7.0, 8.0, 9.0, 10.0])
        print("Matrix is: ")
        print(mat)  # print the confusion matrix 
def main():

	file_name = "voting.nums.csv"

	f = open(file_name)
	raw_data = np.loadtxt(fname=f, delimiter = ',')
	
	data = raw_data[0:]
	X = data[:,1:17]
	y = data[:,0]
	from sklearn.preprocessing import OneHotEncoder
	encoder = OneHotEncoder()         # use the OneHot encoder
	nX = encoder.fit_transform(X)     # fit the data and do the transformation
	print("X shape is", X.shape)
	print("y shape is", y.shape)

	
	dt1 = tree.DecisionTreeClassifier(criterion='entropy')
	dt2 = tree.DecisionTreeClassifier(criterion='entropy',  # use entropy
	                          max_depth=3,          # set max depth
	                          min_samples_split=25, # don't split if too small
	                          min_samples_leaf=2     # don't allow tiny leaves
	                          )

	model1 = dt1.fit(X, y)
	model2 = dt2.fit(X, y)


	#Visualized Trees
	with open("dt1encoder", 'w') as f:
		f = tree.export_graphviz(model1, out_file=f)
	with open("dt2encoder", 'w') as f:
		f = tree.export_graphviz(model2, out_file=f)
	print("Decision Tree 1")
	confMatrix(model1, X, y)
	print("Cross Val Score: ")
	score = cross_val_score(model1, X, y, cv=5)
	print(score)
	print("Mean Score")
	print(np.mean(score))
	print("Decision Tree 2")
	confMatrix(model2, X, y)
	print("Cross Val Score: ")
	score = cross_val_score(model2, X, y, cv=5)
	print(score)
	print("Mean Score")
	print(np.mean(score))
main()
"""Using the one hot encoder had a pretty minimal effect on the values. Without
any tree pruning it negligibly improved the accuracy, but not enough that it 
would come anywhere close to falling outside of eachothers confidence intervals
In decision tree with pruning there was zero effect.
"""
