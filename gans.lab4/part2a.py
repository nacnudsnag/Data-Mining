from sklearn import tree                               # to do DTs
from sklearn.datasets import fetch_mldata              # to import data

iris = fetch_mldata('iris', data_home='.')   # read in the iris dataset

dt = tree.DecisionTreeClassifier(criterion='entropy', # set classifier, use Entropy
				max_depth=3) #set max depth to 3
# I decided to test reducing max_depth and also setting min samples per leaves because both could be potential methods of reducing overfitting, and simplifying the tree.

model = dt.fit(iris.data, iris.target)                  # construct a tree

with open("irisA.dot", 'w') as f:
    f = tree.export_graphviz(model, out_file=f) #creates a .dot file of tree


#When the maximum depth was set to three, the tree obviously became simpler and a little shorter. However, at many of the leaves at the end, the tree did not completely determine which type the iris was. However, although it did not work as well on the training data given, there is a chance it would work better on test data. Either way, this reduced overfitting with the cost of it not being 100% accurate in all situations.
