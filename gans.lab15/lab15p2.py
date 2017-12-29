import matplotlib.pyplot as plt
import pandas
import numpy as np


def main():
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases"
	url += "/pima-indians-diabetes/pima-indians-diabetes.data"
	
	# these come from reading the names file; could be first line in some datasets    
	cols = ['pregnancies', 'plasma', 'bp', 'skinfold', 'insulin', 'bmi',
	        'pedigree', 'age', 'class']
	
	# this adds the column headers to the data frame while reading in the data        
	data = pandas.read_csv(url, names=cols)
	data.hist()   # histogram                                                         
	plt.show()

	data.plot(kind='box', subplots=True, layout=(3,3), sharex=False,
		sharey=False)
	plt.show()

	correlations = data.corr()

	# plot correlation matrix                                                         
	fig = plt.figure()
	ax = fig.add_subplot(111)   # secret code for "1x1 plot; first subplot"
	
	# matshow is the matplotlib command to draw a matrix
	cax = ax.matshow(correlations, vmin=-1, vmax=1)   # range from [-1..1]
	fig.colorbar(cax)                                 # add a legend
	
	# labeling the axes
	ticks = np.arange(0,9,1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(cols)
	ax.set_yticklabels(cols)
	plt.show()
		

	from pandas.tools.plotting import scatter_matrix

	scatter_matrix(data)
	plt.show()

main()
