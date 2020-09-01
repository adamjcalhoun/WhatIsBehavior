import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from prince import MCA
import re

# import umap
import seaborn as sns

# USE MCA INSTEAD OF PCA
# https://github.com/esafak/mca
# http://vxy10.github.io/2016/06/10/intro-MCA/
# https://github.com/MaxHalford/prince#multiple-correspondence-analysis-mca
# https://stackoverflow.com/questions/48521740/using-mca-package-in-python

# NOTE:
# Under the hood Prince uses a randomised version of SVD. This is much faster than using the more commonly full approach. However the results may have a small inherent randomness. For most applications this doesn't matter and you shouldn't have to worry about it. However if you want reproducible results then you should set the random_state parameter.

# conda environment wib

# if MCA has >> 1 relevant dimension, should we do MCA then UMAP to put into a lower dimension?

survey_file = 'C:/Users/adamc/Downloads/wib.csv'
survey_file = '/Users/adamjc/Downloads/wib.csv'

def load_questionnaire(fname):
	# remember to delete the first few rows - these were just me testing the survey
	return pd.read_csv(fname, sep=',')

def filter_questionnaire(df):
	return df[df['Finished'] == 'True']

# what method can provide us with the dimensions of maximum separation between categories?
# SVM?

def ynm_to_dig(df):
	ynm = {'yes':1, 'maybe':0, 'no':-1, 'Yes':1, 'Maybe':0, 'No':-1}

	column_names = ['Q' + str(i+2) for i in range(48)]
	for col_name in column_names:
		df[col_name] = [ynm[item] for item in df[col_name]]

	return df

def dim_reduction(df,dim_type,with_components=False):
	# should we be z-scoring each answer somehow?
	

	if dim_type.lower() == 'pca':
		df = ynm_to_dig(df)
		data = np.array(df)

		pca = PCA(n_components=2)
		pca.fit(data)
		if with_components:
			return pca.transform(data), pca.components_.T
		else:
			return pca.transform(data)
	elif dim_type.lower() == 'mca':
		mca = MCA(n_components=30,benzecri=True)
		mca.fit(df)
		# print(mca.explained_inertia_)
		# print(mca.eigenvalues_) # use Benzecri correction? http://vxy10.github.io/2016/06/10/intro-MCA/#:~:text=Benz%C3%A9cri%20correction,K%2F(K%2D1).
		# print(mca.column_coordinates(df)) # this is the same as 
		# plt.plot(mca.eigenvalues_/np.sum(mca.eigenvalues_),'bo')
		# plt.show()
		if with_components:
			return mca.transform(df), mca.column_coordinates(df)
		else:
			return mca.transform(df)
	elif dim_type.lower() == 'tsne':
		df = ynm_to_dig(df)
		data = np.array(df)

		embedded = TSNE(n_components=2).fit_transform(data)
		return embedded
	# elif dim_type.lower() == 'umap':
	# 	reducer = umap.UMAP()
	# 	embedded = reducer.fit_transform(data)
	# 	return embedded



def plot_PCA_by_field(dim_1=0,dim_2=1):
	# we really want to be using MCA (Multiple correspondence analysis)
	# first load in the questionnaire and filter by responses
	df = load_questionnaire(survey_file)
	df = filter_questionnaire(df)

	
	
	column_names = ['Q' + str(i+2) for i in range(48)]
	question_dim = dim_reduction(df[column_names],'mca')
	question_dim = np.array(question_dim)

	academic_fields = ['Neuroscience','Psychology','Biology','Philosophy','Sociology','Engineering','Medicine','History', 'Languages and Literature', 'Machine Learning', 'Mathematics', 'Statistics', 'Engineering', 'Ethology', 'Ecology']
	plt.figure(figsize=(15,15))

	for fld in academic_fields:
		subj = np.array(df['Q57'].apply(lambda x: pd.Series(x.split(',')).isin([fld]).any()))
		plt.errorbar(np.mean(question_dim[subj,dim_1]),np.mean(question_dim[subj,dim_2]),xerr=np.std(question_dim[subj,dim_1])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,dim_2])/np.sqrt(np.sum(subj)))

	plt.legend(academic_fields)
	plt.xlabel('Factor ' + str(dim_1))
	plt.ylabel('Factor ' + str(dim_2))
	plt.savefig('figs/mca_all_fields.pdf')
	# plt.show()

	plt.figure(figsize=(15,15))
	academic_fields = ['Neuroscience','Psychology','Biology']
	for fld in academic_fields:
		subj = np.array(df['Q57'].apply(lambda x: pd.Series(x.split(',')).isin([fld]).any()))
		plt.errorbar(np.mean(question_dim[subj,dim_1]),np.mean(question_dim[subj,dim_2]),xerr=np.std(question_dim[subj,dim_1])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,dim_2])/np.sqrt(np.sum(subj)))

	sub_fields = [['systems','circuit','circuits'],['cognitive','cognition'],['computational','theoretical','theory'],['molecular','cellular']]
	sub_field_names = ['systems+circuits','cognitive','computational','molecular']
	df['Q57_6_TEXT'] = df['Q57_6_TEXT'].astype(str)
	for fld in sub_fields:
		subj = np.array(df['Q57_6_TEXT'].apply(lambda x: pd.Series(re.split(',|/|and',x.lower().strip())).isin(fld).any()))
		# print((np.mean(question_dim[subj,0]),np.mean(question_dim[subj,1]),np.std(question_dim[subj,0])/np.sqrt(np.sum(subj)),np.std(question_dim[subj,1])/np.sqrt(np.sum(subj))))
		plt.errorbar(np.mean(question_dim[subj,dim_1]),np.mean(question_dim[subj,dim_2]),xerr=np.std(question_dim[subj,dim_1])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,dim_2])/np.sqrt(np.sum(subj)))

	plt.legend(academic_fields + sub_field_names)
	plt.xlabel('Factor ' + str(dim_1))
	plt.ylabel('Factor ' + str(dim_2))
	plt.savefig('figs/mca_subfields.pdf')
	# plt.show()

def plot_PCA_by_animal():
	# we really want to be using MCA (Multiple correspondence analysis)
	# first load in the questionnaire and filter by responses
	df = load_questionnaire(survey_file)
	df = filter_questionnaire(df)

	animals = ['rodents','fish','humans','drosophila','birds','in silico','c. elegans','other invertebrates','other mammals','other non-mammalian vertebrates','other non-animals']
	column_names = ['Q' + str(i+2) for i in range(48)]
	question_dim = dim_reduction(df[column_names],'mca')
	question_dim = np.array(question_dim)

	plt.figure(figsize=(15,15))
	for fld in animals:
		subj = np.array(df['Q58'].apply(lambda x: pd.Series(x.lower().split(',')).isin([fld]).any()))
		plt.errorbar(np.mean(question_dim[subj,0]),np.mean(question_dim[subj,1]),xerr=np.std(question_dim[subj,0])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,1])/np.sqrt(np.sum(subj)))

	plt.legend(animals)
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.savefig('figs/mca_animals.pdf')
	# plt.show()


def plot_PCA_by_seniority():
	# we really want to be using MCA (Multiple correspondence analysis)
	# first load in the questionnaire and filter by responses
	df = load_questionnaire(survey_file)
	df = filter_questionnaire(df)

	seniority = ['professor','postdoc','grad student','undergraduate']
	column_names = ['Q' + str(i+2) for i in range(48)]
	question_dim = dim_reduction(df[column_names],'mca')
	question_dim = np.array(question_dim)

	plt.figure(figsize=(15,15))
	for fld in seniority:
		subj = np.array(df['Q56'].apply(lambda x: pd.Series(x.lower().split(',')).isin([fld]).any()))
		plt.errorbar(np.mean(question_dim[subj,0]),np.mean(question_dim[subj,1]),xerr=np.std(question_dim[subj,0])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,1])/np.sqrt(np.sum(subj)))

	plt.legend(seniority)
	plt.xlabel('Factor 1')
	plt.ylabel('Factor 2')
	plt.savefig('figs/mca_seniority.pdf')
	# plt.show()


def plot_PCA_loadings():
	df = load_questionnaire(survey_file)
	save_df = df
	df = filter_questionnaire(df)

	column_names = ['Q' + str(i+2) for i in range(48)]
	question_dim, components_ = dim_reduction(df[column_names],'mca',with_components=True)

	# print(pd.get_dummies(df[column_names]))
	# print(components_.index.values)
	# print(np.array(components_))
	# exit()

	row_names = components_.index.values
	components_ = np.array(components_)
	plt.plot(components_[:,:3],'.')
	plt.legend(['Factor 1', 'Factor 2', 'Factor 3'])
	plt.xlabel('question')
	plt.ylabel('loading')
	plt.savefig('figs/mca_loadings.pdf')
	# plt.show()

	print('top 3 and bottom 3 questions from Factor 1:')
	for elm in (-components_[:,0]).argsort()[:3]:
		question = row_names[elm].split('_')[0]
		answer = row_names[elm].split('_')[1]
		print(answer + ':    ' + save_df[question].iloc[0])
	for elm in (components_[:,0]).argsort()[:3]:
		question = row_names[elm].split('_')[0]
		answer = row_names[elm].split('_')[1]
		print(answer + ':    ' + save_df[question].iloc[0])

	print('top 3 and bottom 3 questions from Factor 2:')
	for elm in (-components_[:,1]).argsort()[:3]:
		question = row_names[elm].split('_')[0]
		answer = row_names[elm].split('_')[1]
		print(answer + ':    ' + save_df[question].iloc[0])

	for elm in (components_[:,1]).argsort()[:3]:
		question = row_names[elm].split('_')[0]
		answer = row_names[elm].split('_')[1]
		print(answer + ':    ' + save_df[question].iloc[0])

	print('top 3 and bottom 3 questions from Factor 3:')
	for elm in (-components_[:,2]).argsort()[:3]:
		question = row_names[elm].split('_')[0]
		answer = row_names[elm].split('_')[1]
		print(answer + ':    ' + save_df[question].iloc[0])
		
	for elm in (components_[:,2]).argsort()[:3]:
		question = row_names[elm].split('_')[0]
		answer = row_names[elm].split('_')[1]
		print(answer + ':    ' + save_df[question].iloc[0])

	# top 3 and bottom 3 questions from PC1:
	# Does a behavior need to be intentional (does every behavior an animal produces have a purpose)?
	# Behaviors are always discrete; you are either performing that behavior or you are not.
	# All behaviors an animal performs can potentially be identified from recorded video data by using a smart enough computer algorithm.
	# Is sweating a behavior?
	# The knee-jerk reflex is when a tap of a hammer results in the leg extending once before coming to rest. Is the knee-jerk reflex in adults a behavior?
	# A person sweats in response to hot air. Is this person behaving?
	# top 3 and bottom 3 questions from PC2:
	# A behavior is always the output of motor activity
	# The knee-jerk reflex is when a tap of a hammer results in the leg extending once before coming to rest. Is the knee-jerk reflex in adults a behavior?
	# A behavior is always potentially measurable
	# An animal hears one sound, then another. In its mind, it compares the two sounds. Is it behaving?
	# Is learning a behavior?
	# Working memory is temporarily holding something in memory. Is working memory a behavior?

def cluster_responses():
	import umap
	df = load_questionnaire(survey_file)
	df = filter_questionnaire(df)

	column_names = ['Q' + str(i+2) for i in range(48)]
	question_dim = dim_reduction(df[column_names],'mca')
	question_dim = np.array(question_dim)

	reducer = umap.UMAP()
	embedded = reducer.fit_transform(question_dim)

	plt.plot(embedded[:,0],embedded[:,1],'.')
	plt.show()

def plot_clustermap():
	df = load_questionnaire(survey_file)
	save_df = df
	df = filter_questionnaire(df)

	column_names = ['Q' + str(i+2) for i in range(48)]
	# should use jaccard or dice for categorical data??
	# sns.clustermap(ynm_to_dig(df[column_names]),metric='jaccard')
	sns.clustermap(ynm_to_dig(df[column_names]),metric='dice')
	plt.savefig('figs/clustermap.pdf')
	# plt.show()

def get_sub_field():
	# Neurosience: Q57_6_TEXT
	sub_fields = ['systems','circuit','circuits','behavior','cognitive','computational','neuroethology','molecular','cellular',
				  'theoretical','imaging','biophysics','cognition','memory','visual','vision',]
	# Physics: Q57_7_TEXT
	sub_fields = ['biophysics','biological physics']
	# Psychology: Q57_19_TEXT
	sub_fields = ['biological','cognitive','comparative','experimental','clinical','cognition','learning','social',
				  'psychophysics','perception']
	# model systems: Q58
	model_system = ['rodents','fish','humans','drosophila','birds','in silico','c. elegans','other invertebrates','other mammals','other non-mammalian vertebrates','other non-animals']

def save_latex_chart():
	# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_latex.html
	pass

def most_in_consistent_resp():
	# we want to find the responses with the highest % Y/N/M
	# and then with the lowest (highest entropy?)
	df = load_questionnaire(survey_file)
	save_df = df
	df = filter_questionnaire(df)

	column_names = ['Q' + str(i+2) for i in range(48)]
	df = ynm_to_dig(df[column_names])
	data = np.array(df)

	yes_responses = np.mean(data == 1,axis=0)
	no_responses = np.mean(data == -1,axis=0)
	maybe_responses = np.mean(data == 0,axis=0)

	print('most consistently "yes" response:')
	for elm in yes_responses.argsort()[::-1][:5]:
		question = 'Q' + str(elm+2)
		print(question + str((yes_responses[elm],maybe_responses[elm],no_responses[elm])) + ':    ' + save_df[question].iloc[0])

	print('most consistently "no" response:')
	for elm in no_responses.argsort()[::-1][:5]:
		question = 'Q' + str(elm+2)
		print(question + str((yes_responses[elm],maybe_responses[elm],no_responses[elm])) +  ':    ' + save_df[question].iloc[0])

	print('most consistently "maybe" response:')
	for elm in maybe_responses.argsort()[::-1][:5]:
		question = 'Q' + str(elm+2)
		print(question + str((yes_responses[elm],maybe_responses[elm],no_responses[elm])) +  ':    ' + save_df[question].iloc[0])

	print('most inconsistent responses:')
	ent_responses = yes_responses*np.log(yes_responses) + no_responses*np.log(no_responses) + maybe_responses*np.log(maybe_responses)
	for elm in ent_responses.argsort()[:5]:
		question = 'Q' + str(elm+2)
		print(question + str((yes_responses[elm],maybe_responses[elm],no_responses[elm])) +  ':    ' + save_df[question].iloc[0])


def consistency_of_responses():
	# plot cumulative histogram of responses by highest percent answer (Y/N/M)
	# and then print out how many are >80% in agreement

	df = load_questionnaire(survey_file)
	df = filter_questionnaire(df)

	column_names = ['Q' + str(i+2) for i in range(48)]
	df = ynm_to_dig(df[column_names])
	data = np.array(df)

	yes_responses = np.mean(data == 1,axis=0)
	no_responses = np.mean(data == -1,axis=0)
	maybe_responses = np.mean(data == 0,axis=0)

	agreement_level = np.max(np.array([yes_responses,no_responses,maybe_responses]),axis=0)
	hist_steps = np.arange(0,1,0.01)
	pdf,bins = np.histogram(agreement_level,hist_steps)

	plt.step(bins[:-1],np.cumsum(pdf))
	plt.xlabel('highest answer agreement')
	plt.ylabel('cumulative answers')
	plt.savefig('figs/resp_consistency_cdf.pdf')
	# plt.show()

def plot_eigenvalues():
	df = load_questionnaire(survey_file)
	df = filter_questionnaire(df)
	column_names = ['Q' + str(i+2) for i in range(48)]

	mca = MCA(n_components=60,benzecri=True)
	mca.fit(df[column_names])

	plt.plot(mca.eigenvalues_/np.sum(mca.eigenvalues_),'bo')
	plt.xlabel('eigenvalue')
	plt.ylabel('variance explained')
	plt.savefig('figs/mca_eigs.pdf')
	# plt.show()


if __name__ == '__main__':
	# how many PCs do we really need? look at variance explained by number of components
	# for hierarchical clustering: convert to dummy variables and use Jaccard or Dice

	# most_in_consistent_resp()

	# consistency_of_responses()
	# plot_eigenvalues()

	# plot_PCA_loadings()

	# plot_PCA_by_field()
	# plot_PCA_by_animal()
	# plot_PCA_by_seniority()

	plot_clustermap()


	