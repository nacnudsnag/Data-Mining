import numpy as np
import scipy.stats as stats
import math

# data from stats handout... into numpy array data type
# these represent five variations of a system
v1 = np.array([98.5, 97.5, 99.5,99.0,98.0])
v2 = np.array([93.0,100.0, 94.0, 97.5, 99.5])
v3 = np.array([97.0, 98.0, 96.0, 97.5, 96.5])
v4 = np.array([99.3, 98.7, 99.0, 98.5, 98.5])
v5 = np.array([98.0, 99.0, 99.5,100.0, 98.5])

# those will be better as a list, so:
samples = [v1, v2, v3, v4, v5]

def meanCalc(sample):
	means = []
	for each in sample:
		means.append(each.mean())
	return means

sample_means = meanCalc(samples)


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

print(confIntervals(samples))
intervals = confIntervals(samples)
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
fig.savefig("test_conf.png")    
		
