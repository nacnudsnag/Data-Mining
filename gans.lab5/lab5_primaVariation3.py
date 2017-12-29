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
				max_depth=3)	      # set classifier
model = dt.fit(X, y)                                  # construct a tree

with open("iris.dot.Variation3", 'w') as f:
    f = tree.export_graphviz(model, out_file=f)

# in unix:
# dot -Tpdf iris.dot -o iris.pdf
# display iris.pdf &

scores = cross_val_score(dt, X, y, cv=5)
print(scores)

scores2 = cross_val_score(dt, X, y, cv=5)
print(scores2)

#With this variation I only set the max depth to 3. It obviously shrunk and simplified the tree a ton, and actually made it more accurate by about .05 points. It was surprising that making the tree so minimal made it more effective. I would have thought that the extent of the tree would make it more effective, and although I'm sure in some scenarios it did, the overall effect was that it made it less accurate. This was approxamitely equal to the rest of the variations.
