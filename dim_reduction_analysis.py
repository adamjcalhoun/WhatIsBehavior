import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# conda environment wib

def load_questionnaire(fname):
	# remember to delete the first few rows - these were just me testing the survey
	return pd.read_csv(fname, sep=',')

def filter_questionnaire(df):
	return df[df['Finished'].astype(np.int32) == 1]

def dim_reduction(df,dim_type):
	data = np.array(df)

	if dim_type.lower() == 'pca':
		pca = PCA(n_components=2)
		pca.fit(data)
		return pca.transform(data), pca.components_.T

if __name__ == '__main__':
	# first load in the questionnaire and filter by responses
	df = load_questionnaire('C:/Users/adamc/Downloads/wib.csv')
	df = filter_questionnaire(df)

	# now get the questions
	# the first one I forgot to force people to answer....!!!!
	column_names = ['Q' + str(i+2) for i in range(48)]
	question_dim,dims = dim_reduction(df[column_names],'pca')
	neuro = np.array(df['Q57'].str.contains('6'))

	plt.plot(question_dim[neuro,0],question_dim[neuro,1],'.')
	plt.plot(question_dim[np.logical_not(neuro),0],question_dim[np.logical_not(neuro),1],'.')
	plt.show()

	plt.plot(dims)
	plt.show()