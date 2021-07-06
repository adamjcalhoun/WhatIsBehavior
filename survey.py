import h5py
import pandas as pd
import numpy as np
from prince import MCA
import seaborn as sns
import re

from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list
from scipy.cluster import hierarchy
from scipy.spatial import distance

from string import ascii_uppercase

class Survey:
	def __init__(self,filename):
		self.__filename = filename
		self.__column_names = ['Q' + str(i+2) for i in range(48)]

		self.aesthetics = SurveyAesthetics()

		self.load_data()
		self.perform_dim_reduction(self.__finished_df[self.__column_names])

	## loading and preparing the dataset
	def load_data(self):
		# load the data from our file and save the dataframes of all data and filtered data
		self.__original_df = pd.read_csv(self.__filename, sep=',')
		self.filter_questionnaire()
		self.__ynm = self.ynm_to_dig(self.__finished_df[self.__column_names])
		self.__yes_responses = np.mean(self.__ynm == 1,axis=0)
		self.__no_responses = np.mean(self.__ynm == -1,axis=0)
		self.__maybe_responses = np.mean(self.__ynm == 0,axis=0)

	def filter_questionnaire(self):
		column_names = ['Q' + str(i+2) for i in range(48)]
		self.__finished_df = self.__original_df[self.__original_df['Finished'] == 'True']

	def ynm_to_dig(self,df):
		# convert a dataframe with Y/M/N to +1/0/-1
		ynm = {'yes':1, 'maybe':0, 'no':-1, 'Yes':1, 'Maybe':0, 'No':-1}

		ynm_df = df.copy()
		for col_name in self.__column_names:
			ynm_df[col_name] = [ynm[item] for item in ynm_df[col_name]]

		return np.array(ynm_df)


	def get_original_df(self):
		return self.__original_df

	def get_finished_df(self):
		return self.__finished_df

	def get_question_array(self):
		return [self.__original_df[question].iloc[0] for question in self.__column_names]

	## basic yes/no/maybe quantification
	def get_yes_response(self):
		return self.__yes_responses

	def get_no_response(self):
		return self.__no_responses

	def get_maybe_response(self):
		return self.__maybe_responses

	def get_uncertain_response(self):
		ent_responses = self.__yes_responses*np.log(self.__yes_responses) + self.__no_responses*np.log(self.__no_responses) + self.__maybe_responses*np.log(self.__maybe_responses)
		return ent_responses

	def get_most_answer(self,responses):
		resp_list = []
		for elm in responses.argsort()[::-1][:5]:
			question = 'Q' + str(elm+2)
			question = self.aesthetics.format_question(self.__original_df[question].iloc[0])
			resp_list.append([question,self.__yes_responses[elm],self.__maybe_responses[elm],self.__no_responses[elm]])
		
		return resp_list

	def get_answers_from_cluster(self,cluster_num,char_per_line=80):
		resp_list = []

		col_cluster = self.get_col_cluster()
		question_nums = np.where(col_cluster == cluster_num)[0]
		for elm in question_nums:
			question = 'Q' + str(elm+2)
			question = self.aesthetics.format_question(self.__original_df[question].iloc[0],char_per_line=char_per_line)
			resp_list.append([question,self.__yes_responses[elm],self.__maybe_responses[elm],self.__no_responses[elm]])
		
		return resp_list

	def get_ynm(self):
		return self.__ynm


	## write out tables
	def latex_questions_table(self,df,tex_name='figs/question_table.tex'):
		# write out the questions to a tex-readable table
		# columns to write: Question Number, Question, Yes (%), No (%), Maybe (%)
		header = '\\begin{longtable}{p{1cm} p{12cm} || p{1cm} p{1cm} p{1cm}}\n'
		header += '\\caption {Survey questions} \\label{tab:title}\n'
		# header + '\\multicolumn{5}{|c|}{Questions and response data} \\\\ \n'
		# header +  '\\hline \\ \n'
		header += ' & Question & Yes (\\%) & No (\\%) & Maybe (\\%) \\\\ \n'
		header += '\\hline\n'

		ender = '\\hline\n'
		ender += '\\caption{Questions used in this survey and the mean responses.}\n'
		ender += '\\end{longtable}\n'

		questions = self.get_question_array()
		yes = np.around(self.get_yes_response()*100,decimals=2)
		no = np.around(self.get_no_response()*100,decimals=2)
		maybe = np.around(self.get_maybe_response()*100,decimals=2)

		with open(tex_name,'w') as f:
			f.write(header)

			for qq,question in enumerate(questions):
				f.write('Q' + str(qq+1) + ' & ' + question + ' & ' + str(yes[qq]) + ' & ' + str(no[qq]) + ' & ' + str(maybe[qq]) + ' \\\\ \n')

			f.write(ender)

	def latex_metadata_table(self,tex_name='figs/metadata.tex'):
		# https://www.overleaf.com/learn/latex/tables
		# write out the questions to a tex-readable table
		# columns to write: Demographic data, N, %
		header = '\\begin{tabular}{p{12cm} p{2cm} p{2cm}}\n'
		header += '\\caption {Survey metadata} \\label{tab:title}\n'
		header += ' Demographic data & N & \\% \\\\ \n'
		header += '\\hline\n'

		ender = '\\hline\n'
		ender += '\\caption{Metadata of survey respondents.}\n'
		ender += '\\end{tabular}\n'

		with open(tex_name,'w') as f:
			f.write(header)

			# Seniority
			seniority, seniority_names = self.get_seniorities(compressed=False)
			subjects = self.get_field_responses(seniority,'Q56')
			total_subjects = len(subjects[0])
			f.write('Seniority & ' + str(total_subjects) + ' & \\\\ \n')
			for (label,amt) in zip(seniority,subjects):
				f.write('\\hspace{5mm} ' + label[0] + ' & ' + str(sum(amt)) + ' & ' +  str(np.around(sum(amt)/total_subjects*100,decimals=2)) + '\\\\ \n')

			# Gender
			gender = [['Male'],['Female'],['Other'],['Choose not to say']]
			subjects = self.get_field_responses(gender,'Q54')
			total_subjects = len(subjects[0])
			f.write('Gender & ' + str(total_subjects) + ' & \\\\ \n')
			for (label,amt) in zip(gender,subjects):
				f.write('\\hspace{5mm} ' + label[0] + ' & ' + str(sum(amt)) + ' & ' +  str(np.around(sum(amt)/total_subjects*100,decimals=2)) + '\\\\ \n')

			# Academic field
			# -> make sure I'm using all the academic fields
			# -> update data?
			# Biology, Chemistry, Earth science, Ecology, Ethology, Neuroscience, Physics, Political Science,
			# Space sciences, Computer science, Machine Learning, Mathematics, Statistics, Anthropology,
			# Archaeology, Economics, Ethnic studies, Human geography, Psychology, Arts, History, Languages and literature,
			# Law, Philosophy, Sociology, Theology, Business, Engineering, Medicine

			academic_fields, academic_field_names = self.get_academic_fields(compressed=False)
			subjects = self.get_field_responses(academic_fields,'Q57')
			total_subjects = len(subjects[0])
			f.write('Academic fields & ' + str(total_subjects) + ' & \\\\ \n')
			for (label,amt) in zip(academic_fields,subjects):
				f.write('\\hspace{5mm} ' + label[0] + ' & ' + str(sum(amt)) + ' & ' +  str(np.around(sum(amt)/total_subjects*100,decimals=2)) + '\\\\ \n')

			# Model organism
			animals, animal_names = self.get_animals(compressed=False)
			subjects = self.get_field_responses(animals,'Q58')
			total_subjects = len(subjects[0])
			f.write('Model organism & ' + str(total_subjects) + ' & \\\\ \n')
			for (label,amt) in zip(animals,subjects):
				f.write('\\hspace{5mm} ' + label[0] + ' & ' + str(sum(amt)) + ' & ' +  str(np.around(sum(amt)/total_subjects*100,decimals=2)) + '\\\\ \n')


			f.write(ender)


	## dimensionality reduction
	def perform_dim_reduction(self,df):
		self.mca = MCA(n_components=30,benzecri=True)
		self.mca.fit(df)
		self.mca_transformation = self.mca.transform(df)
		self.mca_components = self.mca.column_coordinates(df)

	def get_mca_transformation(self):
		return self.mca_transformation

	def get_mca_components(self):
		return self.mca_components

	def get_mca_eigenvalues(self):
		return self.mca.eigenvalues_

	def regression_predictions(self):
		from sklearn.linear_model import LogisticRegression

		ynm = self.get_ynm()

		# ynm = ynm[:,::-1]

		yes = (ynm == 1).astype(np.int32)
		no = (ynm == -1).astype(np.int32)
		maybe = (ynm == 0).astype(np.int32)

		use_pct = .5
		num_questions = ynm.shape[1]
		train_questions = int(num_questions*use_pct)

		num_folds = 5
		sequence = np.random.permutation(ynm.shape[0])

	
		# not sure these really make a big difference, but I can make some figures to check visually
		regr = LogisticRegression(multi_class='multinomial',max_iter=1000,penalty='elasticnet',solver='saga',l1_ratio=0.5)
		
		df = pd.DataFrame(columns=['Questions','Performance'])

		yes_folds = np.array([yes[sequence[ii::num_folds],train_questions:] for ii in range(num_folds)])
		no_folds = np.array([no[sequence[ii::num_folds],train_questions:] for ii in range(num_folds)])
		maybe_folds = np.array([maybe[sequence[ii::num_folds],train_questions:] for ii in range(num_folds)])
		
		for fold in range(num_folds):
			use_folds = np.array([ii for ii in range(num_folds) if ii is not fold])
			X = np.concatenate((np.concatenate(yes_folds[use_folds],axis=0),
								np.concatenate(no_folds[use_folds],axis=0),
								np.concatenate(maybe_folds[use_folds],axis=0)),axis=1)
			X_test = np.concatenate((yes_folds[fold],no_folds[fold],maybe_folds[fold]),axis=1)

			performance = []
			for Q in range(0,train_questions):
				y = ynm[np.concatenate([sequence[ii::num_folds] for ii in use_folds],axis=0),Q]+1
				regr.fit(X,y)

				y_test = ynm[sequence[fold::num_folds],Q]+1
				y_pred = regr.predict(X_test)

				# can compare to probabilty performance over chance (which is p(y_train=0)*(p(y_test)=0 etc)
				# or just show each prediction separately
				# expected.append(np.sum([np.mean(y == ans)*np.mean(y_test == ans) for ans in range(3)]))

				# performance.append(np.mean(y_pred == y_test))
				df = pd.concat((df,pd.DataFrame([[Q,np.mean(y_pred == y_test)]],columns=['Questions','Performance'])))

			# fold_performance.append(performance)

		yes_folds = np.array([yes[sequence[ii::num_folds],:train_questions] for ii in range(num_folds)])
		no_folds = np.array([no[sequence[ii::num_folds],:train_questions] for ii in range(num_folds)])
		maybe_folds = np.array([maybe[sequence[ii::num_folds],:train_questions] for ii in range(num_folds)])

		
		for fold in range(num_folds):
			use_folds = np.array([ii for ii in range(num_folds) if ii is not fold])
			X = np.concatenate((np.concatenate(yes_folds[use_folds],axis=0),
								np.concatenate(no_folds[use_folds],axis=0),
								np.concatenate(maybe_folds[use_folds],axis=0)),axis=1)
			X_test = np.concatenate((yes_folds[fold],no_folds[fold],maybe_folds[fold]),axis=1)

			performance = []
			for Q in range(train_questions+1,num_questions):
				y = ynm[np.concatenate([sequence[ii::num_folds] for ii in use_folds],axis=0),Q]+1
				regr.fit(X,y)

				y_test = ynm[sequence[fold::num_folds],Q]+1
				y_pred = regr.predict(X_test)

				# can compare to probabilty performance over chance (which is p(y_train=0)*(p(y_test)=0 etc)
				# or just show each prediction separately
				# expected.append(np.sum([np.mean(y == ans)*np.mean(y_test == ans) for ans in range(3)]))
				# performance.append(np.mean(y_pred == y_test))
				df = pd.concat((df,pd.DataFrame([[Q,np.mean(y_pred == y_test)]],columns=['Questions','Performance'])))


			# fold_performance[fold] = fold_performance[fold] + performance

		return df


	## hierarchical clustering
	def cluster_defs(self):
		if not hasattr(self,'__row_linkage'):
			qna = self.get_ynm()

			self.__row_linkage = hierarchy.linkage(distance.pdist(qna), method='ward', metric='euclidean')
			self.__col_linkage = hierarchy.linkage(distance.pdist(qna.T), method='ward', metric='euclidean')

			self.__col_cluster = fcluster(self.__col_linkage,30,'distance')
			self.__row_cluster = fcluster(self.__row_linkage,19,'distance')

	def get_row_cluster(self):
		return self.__row_cluster

	def get_col_cluster(self):
		return self.__col_cluster

	def get_row_linkage(self):
		return self.__row_linkage

	def get_col_linkage(self):
		return self.__col_linkage

	## fields and names
	def get_academic_fields(self,compressed=False):
		if compressed:
			academic_fields = [['Neuroscience','Psychology','Biology','Medicine'],['Engineering','Mathematics', 'Statistics','Machine Learning'],['Ethology', 'Ecology'],['Philosophy','Sociology','History', 'Languages and Literature']]
			academic_field_names = ['Biological Sciences','Math/Engineering','Ecology','Humanities']
			return academic_fields, academic_field_names
		else:
			academic_fields = ['Neuroscience','Psychology','Biology','Medicine','Engineering','Mathematics', 'Statistics','Machine Learning','Ethology', 'Ecology','Philosophy','Sociology','History', 'Languages and Literature']
			return [[a] for a in academic_fields], academic_fields

	def get_academic_field_responses(self,academic_fields):
		subj = []
		for ii,fld in enumerate(academic_fields):
			subj.append(np.array(self.__finished_df['Q57'].apply(lambda x: pd.Series(x.split(',')).isin(fld).any())))

		return subj

	def get_subfields(self,compressed=False):
		sub_fields = [['systems','circuit','circuits'],['cognitive','cognition'],['computational','theoretical','theory'],['molecular','cellular']]
		sub_field_names = ['systems+circuits','cognitive','computational','molecular']
		return sub_fields, sub_field_names

	def get_animals(self,compressed=False):
		if compressed:
			animals = [['humans'],['rodents','other mammals'],['fish','birds','other non-mammalian vertebrates'],['drosophila','c. elegans','other invertebrates'],['in silico']]
			animal_names = ['humans','mammals','other vertebrates','invertebrates','in silico']
			return animals, animal_names
		else:
			animals = ['humans','rodents','other mammals','fish','birds','other non-mammalian vertebrates','drosophila','c. elegans','other invertebrates','in silico']
			return [[a] for a in animals], animals

	def get_seniorities(self,compressed=False):
		if compressed:
			seniority = ['professor','postdoc','grad student','undergraduate']
			return [[s] for s in seniority], seniority
		else:
			seniority = ['professor','postdoc','grad student','undergraduate']
			return [[s] for s in seniority], seniority

	def get_field_responses(self,resp_category,field):
		subj = []

		if 'TEXT' in field:
			# if it is a text field, we need to make sure it is a string type and we will be splitting up the (free response)
			# categories using commas and the word 'and'
			self.__finished_df[field] = self.__finished_df[field].astype(str)
			for ii,fld in enumerate(resp_category):
				subj.append(np.array(self.__finished_df[field].apply(lambda x: pd.Series(re.split(',|/|and',x.lower().strip())).isin(fld).any())))

		else:
			for ii,fld in enumerate(resp_category):
				subj.append(np.array(self.__finished_df[field].apply(lambda x: pd.Series(x.lower().split(',')).isin([f.lower() for f in fld]).any())))

		return subj

	def get_field_response_rate(self,resp_category,field):

		field_metadata = []
	
		if 'TEXT' in field:
			# if it is a text field, we need to make sure it is a string type and we will be splitting up the (free response)
			# categories using commas and the word 'and'
			self.__finished_df[field] = self.__finished_df[field].astype(str)
			for ii,fld in enumerate(resp_category):
				subj = np.array(self.__finished_df[field].apply(lambda x: pd.Series(re.split(',|/|and',x.lower().strip())).isin(fld).any()))
				field_cluster = []
				# field_cluster.append(np.mean(subj))
				for cluster in set(self.__row_cluster):
					field_cluster.append(sum(subj[self.__row_cluster == cluster]) / sum(subj))

				field_metadata.append(field_cluster)

		else:
			for ii,fld in enumerate(resp_category):
				subj = np.array(self.__finished_df[field].apply(lambda x: pd.Series(x.lower().split(',')).isin([f.lower() for f in fld]).any()))
				field_cluster = []
				# field_cluster.append(np.mean(subj))
				for cluster in set(self.__row_cluster):
					field_cluster.append(sum(subj[self.__row_cluster == cluster]) / sum(subj))

				field_metadata.append(field_cluster)

		return field_metadata


class SurveyAesthetics:
	def __init__(self):
		pass

	def format_question(self,question, char_per_line=80):
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

	def plot_dots(self,ax,question_dim,dim_1,dim_2,subj_fields,subj):

		palette_metadata = sns.color_palette('hls',len(subj_fields))
		for ii,fld in enumerate(subj_fields):
			ax.errorbar(np.mean(question_dim[subj[ii],dim_1]),np.mean(question_dim[subj[ii],dim_2]),xerr=np.std(question_dim[subj[ii],dim_1])/np.sqrt(np.sum(subj[ii])),yerr=np.std(question_dim[subj[ii],dim_2])/np.sqrt(np.sum(subj[ii])),
								fmt='o',color=palette_metadata[ii],markeredgecolor='k',ecolor='lightgray',elinewidth=2,capsize=0)


	def clean_dot_plot(self,ax,box_width_scale,xlabel,ylabel,legend_names,font_size,bbox_to_anchor):
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * box_width_scale, box.height])
		# ax.set_position([box.x0, box.y0, box.width*1.5, box.height*1.5])
		ax.legend(legend_names,loc='center left',prop={'size':font_size},bbox_to_anchor=bbox_to_anchor,frameon=False)
		ax.set(xlabel=xlabel,ylabel=ylabel)
		ax.set_ylim(-1*np.max(np.abs(ax.get_ylim())),np.max(np.abs(ax.get_ylim())))
		ax.set_xlim(-1*np.max(np.abs(ax.get_xlim())),np.max(np.abs(ax.get_xlim())))
		ax.spines['left'].set_position('center')
		ax.spines['bottom'].set_position('center')
		ax.spines['right'].set_color('none')
		ax.spines['top'].set_color('none')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_box_aspect(1)

	def YMN_palette(self):
		# https://spectrum.adobe.com/page/color-for-data-visualization/
		# https://seaborn.pydata.org/tutorial/color_palettes.html
		# https://medium.com/nightingale/how-to-choose-the-colors-for-your-data-visualizations-50b2557fa335
		# https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=3
		# https://vega.github.io/vega/docs/schemes/
		# https://carto.com/carto-colors/
		# https://i.imgur.com/e1rBGLT.jpg
		# return sns.color_palette('rocket_r', 3)
		# return sns.color_palette('husl',3)
		# return ['#003f5c','#bc5090','#ffa600']
		# return ['#003f5c','#bc5090','#ffa600']
		return ['#fcca9c','#171717','#b1e3d3']
		# return sns.color_palette('Spectral',3)
		# return sns.color_palette('icefire',10)[::3]
		# return sns.diverging_palette(250, 5, s=100, n=3)
		# ymn suggestions:
		# http://www.sthda.com/english/articles/32-r-graphics-essentials/129-visualizing-multivariate-categorical-data/
		# https://seaborn.pydata.org/generated/seaborn.diverging_palette.html#seaborn.diverging_palette
		# https://seaborn.pydata.org/tutorial/color_palettes.html
		# return sns.diverging_palette(240, 10, s=80, l=30, n=3)


	def draw_clustermap(self,qna,row_cluster,col_cluster,row_linkage,col_linkage):
		col_colors = self.get_clustermap_col_palette(col_cluster)
		row_colors = self.get_clustermap_row_palette(row_cluster)

		cg = sns.clustermap(qna, row_linkage=row_linkage, col_linkage=col_linkage, method="ward", figsize=(8, 8), 
				 			xticklabels=1, yticklabels=0, col_colors=col_colors, row_colors=row_colors,cmap='icefire')
	
		colorbar = cg.ax_heatmap.collections[0].colorbar
		colorbar.set_ticks([0.667,0,-0.667])
		colorbar.set_ticklabels(['Yes', 'Maybe', 'No'])
		cg.ax_heatmap.set_xlabel('questions')
		cg.ax_heatmap.set_ylabel('responses')

		# draw the lines
		tally = np.cumsum([sum(col_cluster == s) for s in set(col_cluster)])
		col_position = np.insert(tally/len(col_cluster),0,[0])
		col_position = col_position[:-1] + (col_position[1:] - col_position[:-1])/2
		for ii in range(len(set(col_cluster))):
			cg.ax_heatmap.plot([tally[ii],tally[ii]],[0,qna.shape[0]],color='1.0',linewidth=4)
			cg.ax_heatmap.text(col_position[ii],1.01,str(ii+1),size=10,color='w',ha='center',transform=cg.ax_heatmap.transAxes)	

		tally = np.cumsum([sum(row_cluster == s) for s in set(row_cluster)])
		row_position = np.insert(tally/len(row_cluster),0,[0])
		row_position = row_position[:-1] + (row_position[1:] - row_position[:-1])/2
		for ii in range(len(set(row_cluster))):
			cg.ax_heatmap.plot([0,qna.shape[1]],[tally[ii],tally[ii]],color='1.0',linewidth=4)
			cg.ax_heatmap.text(-0.035,1-row_position[ii],ascii_uppercase[ii],size=10,color='w',rotation='vertical',va='center',transform=cg.ax_heatmap.transAxes)

		# cg.ax_heatmap.figure.tight_layout()


	def get_clustermap_col_palette(self,col_cluster):
		# color_palette = sns.color_palette('Set2',len(set(col_cluster)))
		color_palette = ['#b3b3b3','#e2e03c','#d46159','#2c72ad','#82bdd5','#3da796']
		return [color_palette[ii-1] for ii in col_cluster]
		
	def get_clustermap_row_palette(self,row_cluster):
		# color_palette = sns.color_palette('Paired',len(set(row_cluster)))
		color_palette = ['#5fb25c','#807e78','#efb71c','#e98640','#b59ad7','#615e99','#96673d']
		return [color_palette[ii-1] for ii in row_cluster]



	def plot_cluster_mean_response(self,ax,response,qna,row_cluster,col_cluster):
		resp_metric = self.get_ymn(response)

		indices = np.arange(0,len(set(col_cluster)))

		category_list = [[np.mean(qna[:,col_cluster == ii] == resp_metric) for ii in set(col_cluster)]]
		for cluster in set(row_cluster):
			qna2 = qna[row_cluster == cluster,:]
			category_list.append([np.mean(qna2[:,col_cluster == ii] == resp_metric) for ii in set(col_cluster)])

		category_data = np.array(category_list)

		category_palette = self.get_category_palette(10)

		hm = sns.heatmap(np.around(category_data,decimals=2),square=True,linewidths=.5,annot=True,ax=ax,cmap=category_palette, 
						xticklabels=[str(ii+1) for ii in range(len(set(col_cluster)))],yticklabels=['all'] + [a for a in ascii_uppercase[:len(set(row_cluster))]],center=0.5, vmin=0, vmax=1,
						cbar_kws={"shrink": .5})

		ax.set_title('mean ' + response + ' responses',fontweight='bold')

	def plot_cluster_relative_response(self,ax,response,qna,row_cluster,col_cluster):
		resp_metric = self.get_ymn(response)

		category_list = [[np.mean(qna[:,col_cluster == ii] == resp_metric) for ii in set(col_cluster)]]
		for cluster in set(row_cluster):
			qna2 = qna[row_cluster == cluster,:]
			category_list.append([np.mean(qna2[:,col_cluster == ii] == resp_metric) for ii in set(col_cluster)])

		category_data = np.array(category_list)

		category_data = category_data / category_data[0]
		category_data = category_data[1:]
		category_data = np.log2(category_data)
		category_data = np.around(category_data,decimals=2)

		category_palette = self.get_relative_palette()

		hm = sns.heatmap(category_data,square=True,linewidths=.5,annot=True,ax=ax,cmap=category_palette,
						xticklabels=[str(ii+1) for ii in range(len(set(col_cluster)))],yticklabels=[a for a in ascii_uppercase[:len(set(row_cluster))]],center=0,vmin=-1,vmax=1,
						cbar_kws={"shrink": .5})

		ax.set_title('relative change in ' + response + ' responses',fontweight='bold')

	def raw_heatmap(self,ax,field_metadata,field_names):
		field_metadata = np.array(field_metadata)
		field_metadata = np.around(field_metadata,decimals=2)
		category_palette = self.get_category_palette(5)

		hm = sns.heatmap(field_metadata.T,square=True,linewidths=.5,annot=True,cmap=category_palette,
						xticklabels=field_names,yticklabels=[a for a in ascii_uppercase[:field_metadata.shape[1]]],ax=ax)
		hm.set_xticklabels(hm.get_xticklabels(), rotation=45,horizontalalignment='right')

	def get_ymn(self,response):
		if response.lower() == 'yes':
			return 1
		elif response.lower() == 'maybe':
			return 0
		elif response.lower() == 'no':
			return -1

	def get_category_palette(self,num_elements):
		return sns.color_palette('rocket_r', num_elements)

	def get_relative_palette(self):
		return sns.color_palette('coolwarm', 3)



