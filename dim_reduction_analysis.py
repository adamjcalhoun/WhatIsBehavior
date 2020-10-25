import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from prince import MCA
import re

from scipy.spatial import distance
from scipy.cluster import hierarchy

# import umap
import seaborn as sns
from matplotlib.colors import ListedColormap

palette_YMN = sns.color_palette('rocket_r', 3)

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
	# sns.clustermap(ynm_to_dig(df[column_names]),metric='dice')

	# need to set: colors by theme (subject, seniority, etc)
	# https://stackoverflow.com/questions/62001483/dendogram-coloring-by-groups
	# also linkage method (single, ward, etc)
	# sns.clustermap((ynm_to_dig(df[column_names])+1)/2,metric='jensenshannon')
	sns.clustermap(ynm_to_dig(df[column_names]),method='ward',metric='euclidean')
	plt.savefig('figs/clustermap.pdf')
	plt.show()

	# then return linkages for later analysis
	# https://stackoverflow.com/questions/27924813/extracting-clusters-from-seaborn-clustermap
	# break down clusters into both question and answer clusters, and report % Y/N/M for answers
	# -> can do major clusters and sub-clusters...
	# in each answer cluster - then give some interpretation
	# also report our metadata: by subfield, by seniority, by model organism

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

def latex_questions_table():
	df = load_questionnaire(survey_file)
	pd.set_option('display.max_colwidth', None)
	with open('figs/question_table.tex','w') as f:
		f.write(df.iloc[0].to_latex(header=True))


def format_question(question, char_per_line=25):

	words = question.split(' ')
	q = ''
	q_tmp = ''
	for ind,w in enumerate(words):
		if len(q_tmp) + len(w) > char_per_line:
			q += q_tmp + '\n'
			q_tmp = w
		else:
			q_tmp += ' ' + w

	q += q_tmp

	return q


def fig1():
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

	fig,axes = plt.subplots(2,2, figsize=(16,8))
	fig.subplots_adjust(hspace=.3,wspace=.3)
	panel_letter_x = -1.5
	panel_letter_y = 1.05
	panel_font_size = 15

	resp_list = []
	for elm in yes_responses.argsort()[::-1][:5]:
		question = 'Q' + str(elm+2)
		question = format_question(save_df[question].iloc[0])
		# print(question + str((yes_responses[elm],maybe_responses[elm],no_responses[elm])) + ':    ' + save_df[question].iloc[0])
		resp_list.append([question,yes_responses[elm],maybe_responses[elm],no_responses[elm]])

	df = pd.DataFrame(resp_list,columns=['Questions','Yes','Maybe','No'])
	df = pd.melt(df,id_vars='Questions',var_name='answers',value_name='response')
	ax = sns.barplot(ax=axes[0,0],x='Questions',y='response',hue='answers',data=df,palette=palette_YMN)
	sns.despine(ax=ax)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=0,horizontalalignment='center',size=6)
	ax.set(ylim=(0, 1),title='most yes responses',xlabel=None)
	ax.text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold')
	
	resp_list = []
	for elm in no_responses.argsort()[::-1][:5]:
		question = 'Q' + str(elm+2)
		question = format_question(save_df[question].iloc[0])
		# print(question + str((yes_responses[elm],maybe_responses[elm],no_responses[elm])) +  ':    ' + save_df[question].iloc[0])
		resp_list.append([question,yes_responses[elm],maybe_responses[elm],no_responses[elm]])

	df = pd.DataFrame(resp_list,columns=['Questions','Yes','Maybe','No'])
	df = pd.melt(df,id_vars='Questions',var_name='answers',value_name='response')
	ax = sns.barplot(ax=axes[0,1],x='Questions',y='response',hue='answers',data=df,palette=palette_YMN)
	sns.despine(ax=ax)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=0,horizontalalignment='center',size=6)
	ax.set(ylim=(0, 1),title='most no responses',xlabel=None)
	ax.text(panel_letter_x,panel_letter_y,'b',size=panel_font_size,weight='bold')

	resp_list = []
	for elm in maybe_responses.argsort()[::-1][:5]:
		question = 'Q' + str(elm+2)
		question = format_question(save_df[question].iloc[0])
		# print(question + str((yes_responses[elm],maybe_responses[elm],no_responses[elm])) +  ':    ' + save_df[question].iloc[0])
		resp_list.append([question,yes_responses[elm],maybe_responses[elm],no_responses[elm]])

	df = pd.DataFrame(resp_list,columns=['Questions','Yes','Maybe','No'])
	df = pd.melt(df,id_vars='Questions',var_name='answers',value_name='response')
	ax = sns.barplot(ax=axes[1,0],x='Questions',y='response',hue='answers',data=df,palette=palette_YMN)
	sns.despine(ax=ax)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=0,horizontalalignment='center',size=6)
	ax.set(ylim=(0, 1),title='most maybe responses',xlabel=None)
	ax.text(panel_letter_x,panel_letter_y,'c',size=panel_font_size,weight='bold')

	resp_list = []
	ent_responses = yes_responses*np.log(yes_responses) + no_responses*np.log(no_responses) + maybe_responses*np.log(maybe_responses)
	for elm in ent_responses.argsort()[:5]:
		question = 'Q' + str(elm+2)
		question = format_question(save_df[question].iloc[0])
		# print(question + str((yes_responses[elm],maybe_responses[elm],no_responses[elm])) +  ':    ' + save_df[question].iloc[0])
		resp_list.append([question,yes_responses[elm],maybe_responses[elm],no_responses[elm]])

	df = pd.DataFrame(resp_list,columns=['Questions','Yes','Maybe','No'])
	df = pd.melt(df,id_vars='Questions',var_name='answers',value_name='response')
	ax = sns.barplot(ax=axes[1,1],x='Questions',y='response',hue='answers',data=df,palette=palette_YMN)
	sns.despine(ax=ax)
	ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='center',size=6)
	ax.set(ylim=(0, 1),title='most disagreement among responses',xlabel=None)
	ax.text(panel_letter_x,panel_letter_y,'d',size=panel_font_size,weight='bold')

	# now plot the questions and the resp
	# plt.tight_layout()
	plt.savefig('figs/fig1_responses.pdf')
	plt.show()

	# labels (for full questions)
	# https://www.drawingfromdata.com/how-to-rotate-axis-labels-in-seaborn-and-matplotlib
	# need to make multi-line labels - can use \n newline
	# rotate labels
	# need to choose colors that are consistent with what we use in other figures

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

def fig2():
	fig,axes = plt.subplots(3,2, figsize=(8,8))
	fig.subplots_adjust(hspace=.5,wspace=.3)

	dim_1=0
	dim_2=2
	panel_letter_x = -1.5
	panel_letter_y = 1.05
	panel_font_size = 15

	# panel 0: MCA loadings
	df = load_questionnaire(survey_file)
	save_df = df
	df = filter_questionnaire(df)

	column_names = ['Q' + str(i+2) for i in range(48)]
	question_dim, components_ = dim_reduction(df[column_names],'mca',with_components=True)
	row_names = components_.index.values

	axes[0,0].axis('off')
	questions = []
	loadings = []
	components_ = np.array(components_)
	loading_srt = components_[:,dim_1].argsort()
	for elm in loading_srt[::-1][:3]:
		questions.append(save_df[row_names[elm].split('_')[0]].iloc[0] + ' ' + row_names[elm].split('_')[1])
		loadings.append(components_[:,dim_1][elm])
	for elm in loading_srt[:3]:
		questions.append(save_df[row_names[elm].split('_')[0]].iloc[0] + ' ' + row_names[elm].split('_')[1])
		loadings.append(components_[:,dim_1][elm])

	axes[0,1].hlines(range(0,-len(loadings),-1),0,loadings,colors='b')
	num_dim1_loadings = len(loadings)

	loadings = []
	loading_srt = components_[:,dim_2].argsort()
	for elm in loading_srt[::-1][:3]:
		questions.append(save_df[row_names[elm].split('_')[0]].iloc[0] + ' ' + row_names[elm].split('_')[1])
		loadings.append(components_[:,dim_2][elm])
	for elm in loading_srt[:3]:
		questions.append(save_df[row_names[elm].split('_')[0]].iloc[0] + ' ' + row_names[elm].split('_')[1])
		loadings.append(components_[:,dim_2][elm])

	axes[0,1].hlines(range(-num_dim1_loadings,-num_dim1_loadings-len(loadings),-1),0,loadings,colors='r')
	# axes[0,0].tick_params(axis='x',labelsize=6)
	axes[0,1].set_yticks(range(0,-num_dim1_loadings-len(loadings),-1))
	axes[0,1].set_yticklabels(questions,size=6)
	# axes[0,0].set(ylim=(0, 1),title='Largest Factor 1 loadings',xlabel=None)
	axes[0,1].set(title='Largest factor loadings',xlabel=None)
	axes[0,1].legend(['Factor 1','Factor 2'],prop={'size':10})
	axes[0,1].text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold')


	# panel 1: by field
	column_names = ['Q' + str(i+2) for i in range(48)]
	# academic_fields = ['Neuroscience','Psychology']
	# academic_fields = ['Sociology']
	academic_fields = ['Neuroscience','Psychology','Biology','Philosophy','Sociology','Engineering','Medicine','History', 'Languages and Literature', 'Machine Learning', 'Mathematics', 'Statistics', 'Engineering', 'Ethology', 'Ecology']

	df = df[df['Q57'].apply(lambda x: pd.Series(x.split(',')).isin(academic_fields).any())]
	# print(df)
	question_dim, components_ = dim_reduction(df[column_names],'mca',with_components=True)
	row_names = components_.index.values


	palette_metadata = sns.color_palette('hls',len(academic_fields))
	ax = axes[1,0]
	question_dim = np.array(question_dim)
	for ii,fld in enumerate(academic_fields):
		subj = np.array(df['Q57'].apply(lambda x: pd.Series(x.split(',')).isin([fld]).any()))
		# print(question_dim.shape)
		# print(subj.shape)
		# print(question_dim[subj,0])
		# exit()
		# axes[1,0].errorbar(np.mean(question_dim[subj,dim_1]),np.mean(question_dim[subj,dim_2]),xerr=np.std(question_dim[subj,dim_1])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,dim_2])/np.sqrt(np.sum(subj)),color=palette_metadata[ii])
		axes[1,0].errorbar(np.mean(question_dim[subj,dim_1]),np.mean(question_dim[subj,dim_2]),xerr=np.std(question_dim[subj,dim_1])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,dim_2])/np.sqrt(np.sum(subj)),
							fmt='o',color=palette_metadata[ii],markeredgecolor='k',ecolor='lightgray',elinewidth=2,capsize=0)

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
	ax.legend(academic_fields,loc='center left',prop={'size':5},bbox_to_anchor=(1,0.5))
	ax.set(xlabel='Factor ' + str(dim_1),ylabel='Factor ' + str(dim_2))
	ax.text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold')


	# panel 2: by subfield

	ax = axes[1,1]
	sub_fields = [['systems','circuit','circuits'],['cognitive','cognition'],['computational','theoretical','theory'],['molecular','cellular']]
	sub_field_names = ['systems+circuits','cognitive','computational','molecular']
	palette_metadata = sns.color_palette('hls',len(sub_fields))
	df['Q57_6_TEXT'] = df['Q57_6_TEXT'].astype(str)
	for ii,fld in enumerate(sub_fields):
		subj = np.array(df['Q57_6_TEXT'].apply(lambda x: pd.Series(re.split(',|/|and',x.lower().strip())).isin(fld).any()))
		# print((np.mean(question_dim[subj,0]),np.mean(question_dim[subj,1]),np.std(question_dim[subj,0])/np.sqrt(np.sum(subj)),np.std(question_dim[subj,1])/np.sqrt(np.sum(subj))))
		ax.errorbar(np.mean(question_dim[subj,dim_1]),np.mean(question_dim[subj,dim_2]),xerr=np.std(question_dim[subj,dim_1])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,dim_2])/np.sqrt(np.sum(subj)),
					fmt='o',color=palette_metadata[ii],markeredgecolor='k',ecolor='lightgray',elinewidth=2,capsize=0)

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
	ax.legend(sub_field_names,loc='center left',prop={'size':5},bbox_to_anchor=(1,0.5))
	ax.set(xlabel='Factor ' + str(dim_1),ylabel='Factor ' + str(dim_2))
	ax.text(panel_letter_x,panel_letter_y,'b',size=panel_font_size,weight='bold')
	

	

	# panel 3: by animal model


	animals = ['rodents','fish','humans','drosophila','birds','in silico','c. elegans','other invertebrates','other mammals','other non-mammalian vertebrates','other non-animals']

	ax = axes[2,0]
	palette_metadata = sns.color_palette('hls',len(animals))
	for ii,fld in enumerate(animals):
		subj = np.array(df['Q58'].apply(lambda x: pd.Series(x.lower().split(',')).isin([fld]).any()))
		ax.errorbar(np.mean(question_dim[subj,dim_1]),np.mean(question_dim[subj,dim_2]),xerr=np.std(question_dim[subj,0])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,1])/np.sqrt(np.sum(subj)),
					fmt='o',color=palette_metadata[ii],markeredgecolor='k',ecolor='lightgray',elinewidth=2,capsize=0)

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
	ax.legend(animals,loc='center left',prop={'size':5},bbox_to_anchor=(1,0.5))
	ax.set(xlabel='Factor ' + str(dim_1),ylabel='Factor ' + str(dim_2))
	ax.text(panel_letter_x,panel_letter_y,'c',size=panel_font_size,weight='bold')

	# panel 4: by seniority
	seniority = ['professor','postdoc','grad student','undergraduate']

	ax = axes[2,1]
	palette_metadata = sns.color_palette('hls',len(seniority))
	for ii,fld in enumerate(seniority):
		subj = np.array(df['Q56'].apply(lambda x: pd.Series(x.lower().split(',')).isin([fld]).any()))
		ax.errorbar(np.mean(question_dim[subj,dim_1]),np.mean(question_dim[subj,dim_2]),xerr=np.std(question_dim[subj,0])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,1])/np.sqrt(np.sum(subj)),
					fmt='o',color=palette_metadata[ii],markeredgecolor='k',ecolor='lightgray',elinewidth=2,capsize=0)

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
	ax.legend(seniority,loc='center left',prop={'size':5},bbox_to_anchor=(1,0.5))
	ax.set(xlabel='Factor ' + str(dim_1),ylabel='Factor ' + str(dim_2))
	ax.text(panel_letter_x,panel_letter_y,'d',size=panel_font_size,weight='bold')

	# plt.tight_layout()
	# plt.show()
	plt.savefig('figs/fig2_mca.pdf')


def fig2_supp1():
	panel_letter_x = -1.5
	panel_letter_y = 1.05
	panel_font_size = 15

	fig,axes = plt.subplots(2,1, figsize=(8,8))
	fig.subplots_adjust(hspace=.3,wspace=.3)

	# Panel 1
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

	axes[0].step(bins[:-1],np.cumsum(pdf))
	axes[0].set(xlabel='highest answer agreement',ylabel='cumulative answers')

	# Panel 2
	df = load_questionnaire(survey_file)
	df = filter_questionnaire(df)
	column_names = ['Q' + str(i+2) for i in range(48)]

	mca = MCA(n_components=60,benzecri=True)
	mca.fit(df[column_names])

	axes[1].plot(mca.eigenvalues_/np.sum(mca.eigenvalues_),'ko')
	axes[1].set(xlabel='eigenvalue',ylabel='variance explained')
	# plt.show()
	plt.savefig('figs/fig2_supp1.pdf')

def fig2_supp2():
	fig,axes = plt.subplots(3,2, figsize=(8,8))
	fig.subplots_adjust(hspace=.3,wspace=.3)

	dim_1=2
	dim_2=3
	panel_letter_x = -1.5
	panel_letter_y = 1.05
	panel_font_size = 15


	# panel 0: MCA loadings
	df = load_questionnaire(survey_file)
	save_df = df
	df = filter_questionnaire(df)

	column_names = ['Q' + str(i+2) for i in range(48)]
	question_dim, components_ = dim_reduction(df[column_names],'mca',with_components=True)

	questions = []
	answers = []
	loading_size = (np.abs(components_[:,dim_1])).argsort()
	for elm in loading_size[:5]:
		questions.append(format_question(row_names[elm].split('_')[1] + ': ' + save_df[row_names[elm].split('_')[0]].iloc[0]))
		loadings.append(components_[:,dim_1][elm])

	axes[0,0].stem(y=answers)
	ax.set_xticklabels(questions, rotation=0,horizontalalignment='center',size=6)
	ax.set(ylim=(0, 1),title='Largest Factor 1 loadings',xlabel=None)
	ax.text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold')

	questions = []
	answers = []
	loading_size = (np.abs(components_[:,dim_2])).argsort()
	for elm in loading_size[:5]:
		questions.append(format_question(row_names[elm].split('_')[1] + ': ' + save_df[row_names[elm].split('_')[0]].iloc[0]))
		loadings.append(components_[:,dim_2][elm])

	axes[0,0].stem(y=answers)
	ax.set_xticklabels(questions, rotation=0,horizontalalignment='center',size=6)
	ax.set(ylim=(0, 1),title='Largest Factor 2 loadings',xlabel=None)
	ax.text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold')

	# panel 1: by field
	df = load_questionnaire(survey_file)
	df = filter_questionnaire(df)

	column_names = ['Q' + str(i+2) for i in range(48)]
	question_dim = dim_reduction(df[column_names],'mca')
	question_dim = np.array(question_dim)

	academic_fields = ['Neuroscience','Psychology','Biology','Philosophy','Sociology','Engineering','Medicine','History', 'Languages and Literature', 'Machine Learning', 'Mathematics', 'Statistics', 'Engineering', 'Ethology', 'Ecology']

	for fld in academic_fields:
		subj = np.array(df['Q57'].apply(lambda x: pd.Series(x.split(',')).isin([fld]).any()))
		axes[1,0].errorbar(np.mean(question_dim[subj,dim_1]),np.mean(question_dim[subj,dim_2]),xerr=np.std(question_dim[subj,dim_1])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,dim_2])/np.sqrt(np.sum(subj)))

	axes[1,0].legend(academic_fields)
	axes[1,0].set(xlabel='Factor ' + str(dim_1),ylabel='Factor ' + str(dim_2))
	axes[1,0].text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold')


	# panel 2: by subfield

	sub_fields = [['systems','circuit','circuits'],['cognitive','cognition'],['computational','theoretical','theory'],['molecular','cellular']]
	sub_field_names = ['systems+circuits','cognitive','computational','molecular']
	df['Q57_6_TEXT'] = df['Q57_6_TEXT'].astype(str)
	for fld in sub_fields:
		subj = np.array(df['Q57_6_TEXT'].apply(lambda x: pd.Series(re.split(',|/|and',x.lower().strip())).isin(fld).any()))
		# print((np.mean(question_dim[subj,0]),np.mean(question_dim[subj,1]),np.std(question_dim[subj,0])/np.sqrt(np.sum(subj)),np.std(question_dim[subj,1])/np.sqrt(np.sum(subj))))
		axes[1,1].errorbar(np.mean(question_dim[subj,dim_1]),np.mean(question_dim[subj,dim_2]),xerr=np.std(question_dim[subj,dim_1])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,dim_2])/np.sqrt(np.sum(subj)))

	axes[1,1].legend(sub_field_names)
	axes[1,1].set(xlabel='Factor ' + str(dim_1),ylabel='Factor ' + str(dim_2))
	axes[1,1].text(panel_letter_x,panel_letter_y,'b',size=panel_font_size,weight='bold')
	

	

	# panel 3: by animal model


	animals = ['rodents','fish','humans','drosophila','birds','in silico','c. elegans','other invertebrates','other mammals','other non-mammalian vertebrates','other non-animals']

	for fld in animals:
		subj = np.array(df['Q58'].apply(lambda x: pd.Series(x.lower().split(',')).isin([fld]).any()))
		axes[2,0].errorbar(np.mean(question_dim[subj,0]),np.mean(question_dim[subj,1]),xerr=np.std(question_dim[subj,0])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,1])/np.sqrt(np.sum(subj)))

	axes[2,0].legend(animals)
	axes[2,0].set(xlabel='Factor ' + str(dim_1),ylabel='Factor ' + str(dim_2))
	axes[2,0].text(panel_letter_x,panel_letter_y,'c',size=panel_font_size,weight='bold')

	# panel 4: by seniority
	seniority = ['professor','postdoc','grad student','undergraduate']

	for fld in seniority:
		subj = np.array(df['Q56'].apply(lambda x: pd.Series(x.lower().split(',')).isin([fld]).any()))
		axes[2,1].errorbar(np.mean(question_dim[subj,0]),np.mean(question_dim[subj,1]),xerr=np.std(question_dim[subj,0])/np.sqrt(np.sum(subj)),yerr=np.std(question_dim[subj,1])/np.sqrt(np.sum(subj)))

	axes[2,1].legend(seniority)
	axes[2,1].set(xlabel='Factor ' + str(dim_1),ylabel='Factor ' + str(dim_2))
	axes[2,1].text(panel_letter_x,panel_letter_y,'d',size=panel_font_size,weight='bold')

	# plt.show()
	plt.savefig('figs/fig2_supp2.pdf')

def fig2_supp3():
	# TODO
	# table with metadata summary
	pass

def fig3():
	# fig.subplots_adjust(hspace=.3,wspace=.3)

	df = load_questionnaire(survey_file)
	save_df = df
	df = filter_questionnaire(df)

	column_names = ['Q' + str(i+2) for i in range(48)]

	# need to set: colors by theme (subject, seniority, etc)
	# https://stackoverflow.com/questions/62001483/dendogram-coloring-by-groups
	# fig = sns.clustermap(ynm_to_dig(df[column_names]),method='ward',metric='euclidean',figsize=(8,8))
	qna = ynm_to_dig(df[column_names])
	row_linkage = hierarchy.linkage(distance.pdist(qna), method='ward', metric='euclidean')
	col_linkage = hierarchy.linkage(distance.pdist(qna.T), method='ward', metric='euclidean')

	# print(qna.shape)
	# print(col_linkage)

	from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list
	# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
	# dend = dendrogram(col_linkage, color_threshold=30)
	# print(fcluster(col_linkage,35,'distance'))
	# print(leaves_list(col_linkage))
	# plt.show()
	# exit()

	# row_colors=network_colors, col_colors=network_colors,
	# https://seaborn.pydata.org/examples/structured_heatmap.html
	# need to make the colormap discrete
	# https://stackoverflow.com/questions/38836154/discrete-legend-in-seaborn-heatmap-plot/56678411
	# also need to make the colormap ticks say 'yes','maybe','no'
	cg = sns.clustermap(qna, row_linkage=row_linkage, col_linkage=col_linkage, method="ward", figsize=(8, 8), xticklabels=1, yticklabels=0)

	# color tree:
	# https://stackoverflow.com/questions/62001483/dendogram-coloring-by-groups

	# colorbar = cg.colorbar(ax.collections[0])
	# colorbar.set_ticklabels(['Yes', 'Maybe', 'No'])
	cg.ax_heatmap.set_xlabel('questions')
	cg.ax_heatmap.set_ylabel('responses')
	# cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xticklabels(), rotation=45,horizontalalignment='right')

	tally = 0
	col_cluster = fcluster(col_linkage,30,'distance')
	for ii in set(col_cluster):
		tally += sum(col_cluster == ii)
		cg.ax_heatmap.plot([tally,tally],[0,qna.shape[0]],color='1.0',linewidth=4)

	tally = 0
	row_cluster = fcluster(row_linkage,19,'distance')
	for ii in set(row_cluster):
		tally += sum(row_cluster == ii)
		cg.ax_heatmap.plot([0,qna.shape[1]],[tally,tally],color='1.0',linewidth=4)



	# https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
	# break down clusters into both question and answer clusters, and report % Y/N/M for answers
	# -> can do major clusters and sub-clusters...
	# https://joernhees.de/blog/tag/hierarchical-clustering/
	# https://stackoverflow.com/questions/28687882/cutting-scipy-hierarchical-dendrogram-into-clusters-via-a-threshold-value

	# plt.savefig('figs/fig3_clustermap.pdf')
	plt.show()
	# exit()

	# then return linkages for later analysis
	# https://stackoverflow.com/questions/27924813/extracting-clusters-from-seaborn-clustermap
	# break down clusters into both question and answer clusters, and report % Y/N/M for answers
	# -> can do major clusters and sub-clusters...
	# in each answer cluster - then give some interpretation
	# also report our metadata: by subfield, by seniority, by model organism

	# responses_by_cluster = []
	qna = np.array(qna)
	# print(qna[:,col_cluster == 1])
	width = 0.2




	# indices = np.arange(0,len(set(col_cluster)))
	# category_list = [np.mean(qna,axis=0)]
	# for cluster in set(row_cluster):
	# 	qna2 = qna[row_cluster == cluster,:]
	# 	category_list.append(np.mean(qna2,axis=0))
	# 	plt.step(range(len(category_list[-1])),category_list[-1])


	# category_data = np.array(y=category_list)
	# category_data = np.log2(category_data / category_data[0])
	# category_data = category_data > 0
	# category_palette = sns.color_palette('rocket_r', 5)
	# plt.step(category_data)
	# sns.heatmap(category_data,square=True,linewidths=.5,annot=True,cmap=category_palette)
	# plt.show()


	fig = plt.figure(constrained_layout=True,figsize=(16,10))
	axes = []
	gs = fig.add_gridspec(2,4)
	axes.append(fig.add_subplot(gs[0,0]))
	axes.append(fig.add_subplot(gs[0,1]))
	axes.append(fig.add_subplot(gs[0,2:]))
	axes.append(fig.add_subplot(gs[1,0]))
	axes.append(fig.add_subplot(gs[1,1:3]))
	axes.append(fig.add_subplot(gs[1,3]))

	# heatmap

	indices = np.arange(0,len(set(col_cluster)))
	category_list = [[np.mean(qna[:,col_cluster == ii] == 1) for ii in set(col_cluster)]]
	for cluster in set(row_cluster):
		qna2 = qna[row_cluster == cluster,:]
		category_list.append([np.mean(qna2[:,col_cluster == ii] == 1) for ii in set(col_cluster)])

	category_data = np.array(category_list)

	category_palette = sns.color_palette('rocket_r', 10)

	ax = axes[0]
	hm = sns.heatmap(category_data,square=True,linewidths=.5,annot=True,ax=ax,cmap=category_palette, center=0.5, vmin=0, vmax=1)

	ax = axes[1]
	category_data = category_data / category_data[0]
	category_data = category_data[1:]
	category_data = np.log2(category_data)
	# category_data = category_data > 0
	category_palette = sns.color_palette('coolwarm', 3)
	# category_palette = sns.cubehelix_palette(3, hue=0.05, rot=0, light=1.0, dark=0)
	hm = sns.heatmap(category_data,square=True,linewidths=.5,annot=True,ax=ax,cmap=category_palette,center=0,vmin=-1,vmax=1)
	# sns.heatmap(category_data,square=True,linewidths=.5,annot=True,cmap=category_palette)
	# plt.show()

	# TODO: add error bars/significance test?



	ax = axes[2]
	academic_fields = ['Neuroscience','Psychology','Biology','Philosophy','Sociology','Engineering','Medicine','History', 'Languages and Literature', 'Machine Learning', 'Mathematics', 'Statistics', 'Engineering', 'Ethology', 'Ecology']
	field_metadata = []
	
	for ii,fld in enumerate(academic_fields):
		subj = np.array(df['Q57'].apply(lambda x: pd.Series(x.split(',')).isin([fld]).any()))
		field_cluster = []
		# field_cluster.append(np.mean(subj))
		for cluster in set(row_cluster):
			field_cluster.append(sum(subj[row_cluster == cluster]) / sum(subj))

		field_metadata.append(field_cluster)

	field_metadata = np.array(field_metadata)
	field_metadata = np.floor(field_metadata*100)/100
	category_palette = sns.color_palette('rocket_r', 5)

	hm = sns.heatmap(field_metadata.T,square=True,linewidths=.5,annot=True,cmap=category_palette,xticklabels=academic_fields,ax=ax)
	hm.set_xticklabels(hm.get_xticklabels(), rotation=45,horizontalalignment='right')
	# plt.show()
	# exit()
	# plt.show()

	ax = axes[3]
	sub_fields = [['systems','circuit','circuits'],['cognitive','cognition'],['computational','theoretical','theory'],['molecular','cellular']]
	sub_field_names = ['systems+circuits','cognitive','computational','molecular']
	df['Q57_6_TEXT'] = df['Q57_6_TEXT'].astype(str)
	field_metadata = []
	for ii,fld in enumerate(sub_fields):
		subj = np.array(df['Q57_6_TEXT'].apply(lambda x: pd.Series(re.split(',|/|and',x.lower().strip())).isin(fld).any()))
		field_cluster = []
		# field_cluster.append(np.mean(subj))
		for cluster in set(row_cluster):
			field_cluster.append(sum(subj[row_cluster == cluster]) / sum(subj))

		field_metadata.append(field_cluster)

	field_metadata = np.array(field_metadata)
	field_metadata = field_metadata
	category_palette = sns.color_palette('rocket_r', 5)
	hm = sns.heatmap(field_metadata.T,square=True,linewidths=.5,annot=True,cmap=category_palette,xticklabels=sub_field_names,ax=ax)
	hm.set_xticklabels(hm.get_xticklabels(), rotation=45,horizontalalignment='right')
	# plt.show()


	ax = axes[4]
	academic_fields = ['rodents','fish','humans','drosophila','birds','in silico','c. elegans','other invertebrates','other mammals','other non-mammalian vertebrates','other non-animals']
	field_metadata = []
	
	for ii,fld in enumerate(academic_fields):
		subj = np.array(df['Q58'].apply(lambda x: pd.Series(x.lower().split(',')).isin([fld]).any()))
		field_cluster = []
		# field_cluster.append(np.mean(subj))
		for cluster in set(row_cluster):
			field_cluster.append(sum(subj[row_cluster == cluster]) / sum(subj))

		field_metadata.append(field_cluster)

	field_metadata = np.array(field_metadata)
	field_metadata = field_metadata
	category_palette = sns.color_palette('rocket_r', 5)
	hm = sns.heatmap(field_metadata.T,square=True,linewidths=.5,annot=True,cmap=category_palette,xticklabels=academic_fields,ax=ax)
	hm.set_xticklabels(hm.get_xticklabels(), rotation=45,horizontalalignment='right')
	# plt.show()

	ax = axes[5]
	academic_fields = ['professor','postdoc','grad student','undergraduate']
	field_metadata = []
	
	for ii,fld in enumerate(academic_fields):
		subj = np.array(df['Q56'].apply(lambda x: pd.Series(x.lower().split(',')).isin([fld]).any()))
		field_cluster = []
		# field_cluster.append(np.mean(subj))
		for cluster in set(row_cluster):
			field_cluster.append(sum(subj[row_cluster == cluster]) / sum(subj))

		field_metadata.append(field_cluster)

	field_metadata = np.array(field_metadata)
	field_metadata = field_metadata
	category_palette = sns.color_palette('rocket_r', 5)
	hm = sns.heatmap(field_metadata.T,square=True,linewidths=.5,annot=True,cmap=category_palette,xticklabels=academic_fields,ax=ax)
	hm.set_xticklabels(hm.get_xticklabels(), rotation=45,horizontalalignment='right')

	# plt.show()
	plt.savefig('figs/fig4_cluster_responses.pdf')



if __name__ == '__main__':
	# how many PCs do we really need? look at variance explained by number of components
	# for hierarchical clustering: convert to dummy variables and use Jaccard or Dice

	# fig1()
	
	# fig2()
	# fig2_supp1()
	# fig2_supp2()
	# fig2_supp3()

	fig3()

	# latex_questions_table()

	# consistency_of_responses()
	# plot_eigenvalues()

	# plot_PCA_loadings()

	# plot_PCA_by_field()
	# plot_PCA_by_animal()
	# plot_PCA_by_seniority()

	# plot_clustermap()


