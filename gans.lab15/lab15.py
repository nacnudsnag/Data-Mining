import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def main():

	url = "https://archive.ics.uci.edu/ml/machine-learning-databases"
	url += "/pima-indians-diabetes/pima-indians-diabetes.data"
	
	# these come from reading the names file; could be first line in some datasets
	cols = ['pregnancies', 'plasma', 'bp', 'skinfold', 'insulin', 'bmi',
	        'pedigree', 'age', 'class']
	
	# this adds the column headers to the data frame while reading in the data
	data = 	pd.read_csv(url, names=cols)
	
	
	# use the column for pregnancies, make three bins, use numbers, not labels
	newcol = pd.qcut(data['pregnancies'], 3, labels=False)
	print(newcol)
	
	data['q_pregnancies'] = newcol
	print(data)


	# set the columns you want, in the order that you want
	cols = ['q_pregnancies', 'plasma', 'bp', 'skinfold', 'insulin', 'bmi',
	'pedigree', 'age', 'class']

	# reassign the data frame (select specific columns)
	data = data[cols]


main()
