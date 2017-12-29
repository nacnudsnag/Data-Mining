from sklearn import tree       # to do DTs
import numpy as np             # to use numpy
from sklearn.datasets import load_digits
import scipy.stats as stats
import math





digits = load_digits()
print("Digits shape is: ", digits.images.shape)

X = digits.data
print("X shape is", X.shape)

y = digits.target
print("y shape is", y.shape)

import matplotlib.pyplot as plt

def plot_data ():
    fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
        ax.text(0.05, 0.05, str(digits.target[i]),
                transform=ax.transAxes, color='green')

    #plt.show()

    # use this to write to a file; look at the file with display
    fig.savefig('test_digits.png')

    return None

#plot_data()

# make decision tree
dt = tree.DecisionTreeClassifier(criterion='entropy') # set classifier
model = dt.fit(X, y)                                  # construct a tree


from sklearn.model_selection import cross_val_score

dtScores = cross_val_score(dt, X, y, cv=5)


## Make Random Forest

from sklearn.ensemble import RandomForestClassifier
def randForest(seed):
	randForest = RandomForestClassifier(n_estimators=100,         #make 100 trees
         	                    random_state=seed,        #seed to random # gen
                	            criterion='entropy')   #use entropy

	scores1 = cross_val_score(dt, X, y, cv=5)
	print(scores1)
	return randForest

rf = randForest(0)
### make X and y training and testing data (one split)
### can change random_state to get different splits                             
originalRF = cross_val_score(rf, X, y, cv = 5)
print(originalRF)
print("ASDFASDFASDFASDFASDFASF")
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def confMatrix(model):
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                test_size=.2,
                                                random_state=10)  
	# use the model on the training data
	# (whatever your model variable was last set to)
	model.fit(Xtrain, ytrain)
	y_model = model.predict(Xtest)


	score = accuracy_score(ytest, y_model)
	print("Score is: ")
	print(score)

	### confusion matrix:
	mat = confusion_matrix(ytest, y_model)
	print("Matrix is: ")
	print(mat)  # print the confusion matrix 


confMatrix(model)
confMatrix(rf)

print("Creating Random Forest 1")
rfVariation1 = randForest(40)
print("Creating Random Forest 2")
rfVariation2 = randForest(51)
print("Creating Random Forest 3")
rfVariation3 = randForest(12)

#print("Matrices")
#print("Variation 1")
#confMatrix(rfVariation1)
#print("Variation 2")
#confMatrix(rfVariation2)
#print("Variation 3")
#confMatrix(rfVariation3)


print("Before Declarations")
RF1 = cross_val_score(rfVariation1, X, y, cv=5)
RF2 = cross_val_score(rfVariation2, X, y, cv=5)
RF3 = cross_val_score(rfVariation3, X, y, cv=5)
variations = [dtScores, originalRF, RF1, RF2, RF3]
print(RF3)
def meanCalc(sample):
        means = []
        for each in sample:
                means.append(each.mean())
        return means

def confIntervals(samples):
	z_critical = stats.norm.ppf(q = 0.975)
	means = meanCalc(samples)
	error = []
	for sample in samples:
		error.append(z_critical * sample.std() / np.sqrt(len(samples)))
	confInterv = []
	i = 0
	while i < len(samples):
		confInterv.append([means[i] - error[0], means[i] + error[0]])
		i+=1
	return confInterv
sample_means = meanCalc(variations)

intervals = confIntervals(variations)
import matplotlib.pyplot as plt

plt.figure(figsize=(9,9))

xvals = np.arange(5, 30, 5)
yerrors = [(top-bot)/2 for top,bot in intervals]
fig = plt.figure()
plt.errorbar(x=xvals,
             y=sample_means,
             yerr=yerrors,
             fmt='D')
plt.show()

# use this to write to a file; look at the file with display  
fig.savefig("variations.png")    
	





