from sklearn import tree                               # to do DTs
from sklearn.datasets import fetch_mldata              # to import data

iris = fetch_mldata('iris', data_home='.')   # read in the iris dataset

dt = tree.DecisionTreeClassifier(criterion='entropy', # set classifier, use Entropy
                                 min_samples_leaf=4) #set min samples per leaf to 4
# I decided to test reducing max_depth and also setting min samples per leaves because both could be potential methods of reducing overfitting, and simplifying the tree.

model = dt.fit(iris.data, iris.target)                  # construct a tree

with open("irisB.dot", 'w') as f:
    f = tree.export_graphviz(model, out_file=f) #creates a .dot file of tree

# When the minimum samples per leaf was set to four, the tree once again became a little simple and shorter. Doing so, reduced some of the accuracy because it overfit less. The depth ended up being about 3, similar to when the max depth was set to three. In one scenario, it ended up splitting a node where the data was 47 of one class and only 1 of another class, which led to a negative infogain. Similar to setting a max depth, it reduced overfitting, with the cost being that it became a little less accurate. Max Depth seemed to work better because it reduced the size and the overfitting, but there wasn't nodes at the end with negative infogain.
