import numpy as np             # to use numpy
import urllib.request          # to retrieve URL files
from sklearn import tree       # to do DTs#URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
from sklearn.model_selection import cross_val_score

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/"
url += "pima-indians-diabetes/pima-indians-diabetes.data"

# download the file
raw_data = urllib.request.urlopen(url)

# load the CSV file as a numpy matrix        
dataset = np.loadtxt(raw_data, delimiter=",")
#print(dataset.shape)   # if you like

# separate the data from the target attributes                             
X = dataset[:,0:7]
y = dataset[:,8]

# make decision tree
dt = tree.DecisionTreeClassifier(criterion='entropy',
				 max_depth=3,          # set max depth
                                 min_samples_split=25, # don't split if too small
                                 min_samples_leaf=2)    # don't allow tiny leaves)
model = dt.fit(X, y)  # construct a tree

with open("iris.dot.Variation1", 'w') as f:
    f = tree.export_graphviz(model, out_file=f)

# in unix:
# dot -Tpdf iris.dot -o iris.pdf
# display iris.pdf &

scores = cross_val_score(dt, X, y, cv=5)
print(scores)

scores2 = cross_val_score(dt, X, y, cv=5)
print(scores2)

#This variation was supposed to combine multiple ways of simplifying the tree in order to reduce overfitting the most. Although it did reduce overfitting and increase the test scores by some margin, the only factor that actually had an impact was setting the max depth to 3. Since the max depth was three, the other to variables never had an impact because the tree never even got to the point where it could get to a leaf with 2 samples, or really split anything into less than 25 samples. This had the exact same impact as variation three in which I only changed the max depth. If I were to increase the min samples per leaf and per split, or increase the max depth, then the other variables could actually have an impact.

