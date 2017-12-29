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
	with open("dt1", 'w') as f:
		f = tree.export_graphviz(model1, out_file=f)
	with open("dt2", 'w') as f:
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
"""Decision tree 1 is very complex. The depth is eleven, and it uses pretty much 
every vote option available. The accuracy ranges everywhere from .93 to .96 so 
is relatively accurate. The main variable that accounts for the split is
attribute 3, or the third vote. In looking at the UCI database online, the 
third attribute was on budget resolution. It makes sense that this could be an
indicator of partisanship. This attribute is at the top of the tree and reduces the 
entropy significantly. If the vote is no, the entropy reduces to .06, and if 
the vote is yes, it still halves to .5. 
Decision tree 2 is obviously simpler. The depth is limited to just three. 
However, because there is now less overfitting the scores are actually 
potentially higher. The scores range everywhere from .918 to .988, so in all
the accuracy is pretty much the same. Once again, the main attribute that
is split on is vote 3, or the 3rd non class attribute.

"""

