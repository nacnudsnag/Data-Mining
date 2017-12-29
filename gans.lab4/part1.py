from sklearn import tree                               # to do DTs
from sklearn.datasets import fetch_mldata              # to import data

iris = fetch_mldata('iris', data_home='.')   # read in the iris dataset

dt = tree.DecisionTreeClassifier(criterion='entropy') # set classifier
model = dt.fit(iris.data, iris.target)                  # construct a tree

with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(model, out_file=f)	#creates a .dot file of tree

