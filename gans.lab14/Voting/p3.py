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

	import pandas as pd
	filename = "voting.csv"              #not converted to numbers
	df = pd.read_csv(filename,sep=',')   # read into a pandas data frame
	
	from sklearn.preprocessing import LabelEncoder
	encoder = LabelEncoder()             # make the encoder
	df = df.apply(encoder.fit_transform) # apply it to all the items in the data frame
	
	votes = np.array(df)                 # convert to numpy array
	
	#... and go on your way
	X = votes[:,1:16]
	y = votes[:,0]

	#print(X.shape)
	#print(y.shape)

	dt1 = tree.DecisionTreeClassifier(criterion='entropy')
	dt2 = tree.DecisionTreeClassifier(criterion='entropy',  # use entropy
                                  max_depth=3,          # set max depth
                                  min_samples_split=25, # don't split if too small
                                  min_samples_leaf=2     # don't allow tiny leaves
                                  )

	model1 = dt1.fit(X, y)
	model2 = dt2.fit(X, y)
	#Visualized Trees
	with open("dt1AlternateData", 'w') as f:
		f = tree.export_graphviz(model1, out_file=f)
	with open("dt2AlternateData", 'w') as f:
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

"""Reading the data in in a different way, did not have a huge impact on the actual scores
it could have changed the scores some negligible amount, but not enought that I could tell.
However, the confusion matrices were different. Instead of having the clustering of data 
in the middle, the scores in the given rows and columns were centered in the top left side
where col=0 and row=0. It didn't have an impact on the scores of the learned model significantly
but it could have changed the odel and the specific way it was built.
"""
