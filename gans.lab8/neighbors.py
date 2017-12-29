import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
import matplotlib.pyplot as plt
def main():
	digits = load_digits()
	print("Digits shape is: ", digits.images.shape)
	
	X = digits.data
	print("X shape is", X.shape)
	
	y = digits.target
	print("y shape is", y.shape)
	
	# model is knn; k=1
	model = KNeighborsClassifier(n_neighbors=1,
	                             weights='uniform')  #default
	
	# one t-t split (not a cross-validation)
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
	                                                test_size=.2,    # 20%
	                                                random_state=0)  # seed
	# train knn
	model.fit(Xtrain, ytrain)
	# predictions 
	y_model = model.predict(Xtest)
	
	# get the prediction accuracy for this run
	score = accuracy_score(ytest, y_model)
	print("accuracy for one t-t split is", score)
	
	nScores1 = cross_val_score(model, X, y, cv=5)
	print(nScores1)

	nScores3 = nearestNeighborsScores(3, X, y)
	nScores5 = nearestNeighborsScores(5, X, y)
	nScores7 = nearestNeighborsScores(7, X, y)
	nScores9 = nearestNeighborsScores(9, X, y)

	kVariations = [nScores1, nScores3, nScores5, nScores7, nScores9]

	sample_means = meanCalc(kVariations)
	intervals = confIntervals(kVariations)
	
	plt.figure(figsize=(9,9))

	xvals = np.arange(5, 30, 5)
	yerrors = [(top-bot)/2 for top,bot in intervals]
	fig = plt.figure()
	plt.errorbar(x=xvals,
	             y=sample_means,
	             yerr=yerrors,
	             fmt='D')
	labels = ["k=1", "k=3", "k=5", "k=7", "k=9"]
	plt.xticks(xvals,labels)

	# use this to write to a file; look at the file with display  
	fig.savefig("kvariations.png")
	for i in range(0, 5):
		for j in range(i + 1, 5):
			printPs(i, j, kVariations)

def nearestNeighborsScores(k, X, y):
	model = KNeighborsClassifier(n_neighbors=k,
				     weights='uniform') #default
	return cross_val_score(model, X, y, cv=5)


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
def printPs(sample, sample2, variations):
	pair = stats.ttest_rel(variations[sample], variations[sample2])
	print("The p-value for {0}, {1} is {2}".format(sample, sample2, pair[1]))	
main()
		
