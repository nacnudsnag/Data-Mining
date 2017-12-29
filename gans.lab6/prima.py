import numpy as np             # to use numpy
import urllib.request          # to retrieve URL files
from sklearn import tree       # to do DTs#URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
import math


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
dt = tree.DecisionTreeClassifier(criterion='entropy') # set classifier
model = dt.fit(X, y)                                  # construct a tree

prima = cross_val_score(dt, X, y, cv=5)


dt1 = tree.DecisionTreeClassifier(criterion='entropy',
				 max_depth=3,         
                                 min_samples_split=25,
                                 min_samples_leaf=2) 
model = dt1.fit(X, y)

prima1 = cross_val_score(dt1, X, y, cv=5)

dt2 = tree.DecisionTreeClassifier(criterion='entropy',
                                 min_samples_leaf=2)
model = dt2.fit(X, y)

prima2 = cross_val_score(dt2, X, y, cv=5)

dt3 = tree.DecisionTreeClassifier(criterion='entropy',
                                 min_samples_split=40)
model = dt3.fit(X, y)

prima3 = cross_val_score(dt3, X, y, cv=5)

variations = [prima, prima1, prima2, prima3]

def meanCalc(sample):
	means = []
	for each in sample:
		means.append(each.mean())
	return means

sample_means = meanCalc(variations)


def confIntervals(samples):
	z_critical = stats.norm.ppf(q = 0.975)
	means = sample_means
	error = []
	for sample in samples:
		error.append(z_critical * sample.std() / np.sqrt(len(samples)))
	confInterv = []
	i = 0
	while i < len(samples):
		confInterv.append([means[i] - error[0], means[i] + error[0]])
		i+=1
	return confInterv

print(confIntervals(variations))
intervals = confIntervals(variations)
import matplotlib.pyplot as plt

plt.figure(figsize=(9,9))

xvals = np.arange(5, 25, 5)
yerrors = [(top-bot)/2 for top,bot in intervals]
fig = plt.figure()
plt.errorbar(x=xvals,
             y=sample_means,
             yerr=yerrors,
             fmt='D')
plt.show()

# use this to write to a file; look at the file with display  
fig.savefig("variations.png")    
	
