import matplotlib.pyplot as plt
from survey import Survey

from scipy.stats import pearsonr

import pandas as pd
import seaborn as sns
import numpy as np

survey_file = 'wib_clean.csv'

# let's break up figure 1 into two figures, one with the example diversity in questions
# and one with the regression prediction (showing that even though people don't agree, they are predictable)
# this should be both the result overall and an example prediction (show largest regression coefficients)

def fig1(behavior_survey,show_plot=False,savefilename=None):
	# example questions,
	# prediction of regression model?
	# MCA analysis
	# fig,axes = plt.subplots(4,2, figsize=(9,8))

	fig = plt.figure(figsize=(8, 10))

	gs = fig.add_gridspec(4,2, left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
	ax = fig.add_subplot(gs[1, 0])
	
	# fig.subplots_adjust(hspace=.5,wspace=.3)

	dim_1=0
	dim_2=1
	panel_letter_x = -0.4
	panel_letter_y = 1.1
	panel_font_size = 15
	bbox_to_anchor = (1,0.5)
	legend_font_size = 8
	box_width_scale = 1.0


	# axes[0,0].axis('off')
	fig.add_subplot(gs[0, 0]).axis('off')
	ax = fig.add_subplot(gs[0, 1])
	resp_list = behavior_survey.get_most_answer(-1*behavior_survey.get_uncertain_response())

	# ax = axes[0,1]
	df = pd.DataFrame(resp_list,columns=['Questions','Yes','Maybe','No'])
	df = pd.melt(df,id_vars='Questions',var_name='answers',value_name='response')
	ax = sns.barplot(ax=ax,y='Questions',x='response',hue='answers',data=df,palette=behavior_survey.aesthetics.YMN_palette(),linewidth=1,edgecolor='0')
	# sns.despine(ax=ax)

	ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center',size=8)
	ax.set(xlim=(0, 1),title='responses with most disagreement',xlabel='Proportion of answers')
	ax.text(panel_letter_x-1.1,panel_letter_y,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)
	ax.spines['left'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')


	# regression predictions
	# axes[1,0]; axes[1,1]

	regr_perf = behavior_survey.regression_predictions()
	# print(regr_perf)
	# print(regr_perf['Performance'].mean())
	ax = fig.add_subplot(gs[1, :])
	sns.stripplot(x='Questions',y='Performance',data=regr_perf,alpha=.25,color='k',ax=ax)
	sns.pointplot(x='Questions',y='Performance',data=regr_perf,alpha=.25,join=False,scale=0.5,color='k',markers='d',ci=None,ax=ax)
	plt.plot([-1,47.5],[1/3,1/3],'k--')
	ax.set(ylim=(0,1),xlim=(-1,47.5))
	ax.set(xticklabels=[])
	ax.set_xticks([])
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.text(panel_letter_x,panel_letter_y,'b',size=panel_font_size,weight='bold',transform=ax.transAxes)


	# ax = axes[1,1]
	# ax.axis('off')

	question_dim = np.array(behavior_survey.get_mca_transformation())


	# panel 1: by field	
	# ax = axes[2,0]
	ax = fig.add_subplot(gs[2, 0])
	academic_fields, academic_field_names = behavior_survey.get_academic_fields(compressed=True)
	subjects = behavior_survey.get_field_responses(academic_fields,'Q57')
	behavior_survey.aesthetics.plot_dots(ax,question_dim,dim_1,dim_2,academic_fields,subjects)
	# behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'Factor ' + str(dim_1),'Factor ' + str(dim_2),academic_field_names,legend_font_size,bbox_to_anchor)
	behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'','',academic_field_names,legend_font_size,bbox_to_anchor)
	ax.text(panel_letter_x,panel_letter_y,'c',size=panel_font_size,weight='bold',transform=ax.transAxes)


	# panel 2: by subfield
	# ax = axes[2,1]
	ax = fig.add_subplot(gs[2, 1])
	sub_fields, sub_field_names = behavior_survey.get_subfields()
	subjects = behavior_survey.get_field_responses(sub_fields,'Q57_6_TEXT')
	behavior_survey.aesthetics.plot_dots(ax,question_dim,dim_1,dim_2,sub_fields,subjects)
	# behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'Factor ' + str(dim_1),'Factor ' + str(dim_2),sub_field_names,legend_font_size,bbox_to_anchor)
	behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'','',sub_field_names,legend_font_size,bbox_to_anchor)
	ax.text(panel_letter_x,panel_letter_y,'d',size=panel_font_size,weight='bold',transform=ax.transAxes)

	# panel 3: by animal model
	# ax = axes[3,0]
	ax = fig.add_subplot(gs[3, 0])
	animals, animal_names = behavior_survey.get_animals(compressed=True)
	subjects = behavior_survey.get_field_responses(animals,'Q58')
	behavior_survey.aesthetics.plot_dots(ax,question_dim,dim_1,dim_2,animals,subjects)
	# behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'Factor ' + str(dim_1),'Factor ' + str(dim_2),animal_names,legend_font_size,bbox_to_anchor)
	behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'','',animal_names,legend_font_size,bbox_to_anchor)
	ax.text(panel_letter_x,panel_letter_y,'e',size=panel_font_size,weight='bold',transform=ax.transAxes)

	# panel 4: by seniority
	# ax = axes[3,1]
	ax = fig.add_subplot(gs[3, 1])
	seniority, seniority_names = behavior_survey.get_seniorities(compressed=False)
	subjects = behavior_survey.get_field_responses(seniority,'Q56')
	behavior_survey.aesthetics.plot_dots(ax,question_dim,dim_1,dim_2,seniority,subjects)
	# behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'Factor ' + str(dim_1),'Factor ' + str(dim_2),seniority_names,legend_font_size,bbox_to_anchor)
	behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'','',seniority_names,legend_font_size,bbox_to_anchor)
	ax.text(panel_letter_x,panel_letter_y,'f',size=panel_font_size,weight='bold',transform=ax.transAxes)

	# plt.tight_layout()
	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()

def fig2(behavior_survey,show_plot=False,savefilename=None):
	behavior_survey.cluster_defs()

	behavior_survey.aesthetics.draw_clustermap(behavior_survey.get_ynm(),behavior_survey.get_row_cluster(),behavior_survey.get_col_cluster(),behavior_survey.get_row_linkage(),behavior_survey.get_col_linkage())

	# plt.figure.tight_layout()
	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()

def fig3(behavior_survey,show_plot=False,savefilename=None):
	panel_letter_x = -0.4
	panel_letter_y = 1.1
	panel_font_size = 15


	behavior_survey.cluster_defs()
	num_clusters = 6 # do this programmatically 

	fig,axes = plt.subplots(num_clusters,4,figsize=(12,8))
	# heatmap
	
	# one column is the name of the type of behavior

	# one column are some example survey questions

	# one column is Kathleen's illustrations

	illustration_list = ['cluster_1.png','cluster_2.png','cluster_3.png','cluster_4.png','cluster_5.png','cluster_6.png']
	cluster_name = ['reflex','actions','understanding the mind','motor or sensorimotor','non-animal','cognition']
	question_list = [0,2,2,0,1,1]
	letters = 'abcdef'

	# title on top of each of these
	for behavior_ind in range(num_clusters):
		img = plt.imread('imgs/' + illustration_list[behavior_ind])

		# this should be underlined
		if behavior_ind == 0:
			# bold these
			axes[0,0].set(title='example question')
			axes[0,1].set(title='average response')
			axes[0,2].set(title='definition name')
			axes[0,3].set(title='illustration')


		# this should be separated better; maybe with grey background? or just better whitespace?
		ax = axes[behavior_ind,2]
		ax.text(0.5,0.5,cluster_name[behavior_ind],horizontalalignment='center')
		ax.axis('off')
		# ax.set_facecolor((0.6,0.6,0.6))

		ax = axes[behavior_ind,1]
		# get the questions corresponding to each cluster
		# and just choose the first 3 or 4... or choose representative samples?
		# that's probably easier... just hand-code this.
		# axes[0,0].axis('off')
		resp_list = behavior_survey.get_answers_from_cluster(behavior_ind+1,char_per_line=30)

		df = pd.DataFrame([resp_list[question_list[behavior_ind]]],columns=['Questions','Yes','Maybe','No'])
		df = pd.melt(df,id_vars='Questions',var_name='answers',value_name='response')
		ax = sns.barplot(ax=ax,y='Questions',x='response',hue='answers',data=df,palette=behavior_survey.aesthetics.YMN_palette(),linewidth=1,edgecolor='0')
		ax.set(ylabel=None,xlabel=None,xlim=(0, 1))
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_visible(False)
		
		if behavior_ind != 0:
			ax.legend_.remove()
		else:
			# ax.legend(loc='center left',prop={'size':font_size},bbox_to_anchor=(1,0.5))
			ax.legend(loc='center left',bbox_to_anchor=(0.75,0.5),frameon=False)

		if behavior_ind != num_clusters-1:
			ax.spines['bottom'].set_visible(False)
			ax.set(xticklabels=[])
			ax.set_xticks([])
		else:
			ax.set(xlabel='proportion answers')

		ax = axes[behavior_ind,0]
		ax.text(panel_letter_x,panel_letter_y-0.1,letters[behavior_ind],size=panel_font_size,weight='bold',transform=ax.transAxes)
		ax.axis('off')

		# ax.axis('off')

		ax = axes[behavior_ind,3]
		ax.imshow(img)
		ax.axis('off')


	# axes[0,0].axis('off')
	# resp_list = behavior_survey.get_most_answer(-1*behavior_survey.get_uncertain_response())

	# ax = axes[0,1]
	# df = pd.DataFrame(resp_list,columns=['Questions','Yes','Maybe','No'])
	# df = pd.melt(df,id_vars='Questions',var_name='answers',value_name='response')
	# ax = sns.barplot(ax=ax,y='Questions',x='response',hue='answers',data=df,palette=behavior_survey.aesthetics.YMN_palette())
	# # sns.despine(ax=ax)

	# ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center',size=8)
	# ax.set(xlim=(0, 1),title='responses with most disagreement',xlabel=None)
	# ax.text(panel_letter_x-1.1,panel_letter_y,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)



	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()
		


def fig4(behavior_survey,show_plot=False,savefilename=None):
	panel_letter_x = -0.2
	panel_letter_y = 1.05
	panel_font_size = 15

	fig,axes = plt.subplots(2,2, figsize=(8,8))
	# heatmap

	# ax = axes[0,0]
	# ax.plot(fold_performance)
	# ax.text(panel_letter_x,panel_letter_y-0.1,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)

	ax = axes[0,0]
	behavior_survey.aesthetics.plot_cluster_mean_response(ax,'Yes',behavior_survey.get_ynm(),behavior_survey.get_row_cluster(),behavior_survey.get_col_cluster())
	ax.text(panel_letter_x,panel_letter_y-0.1,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)
	
	ax = axes[0,1]
	behavior_survey.aesthetics.plot_cluster_relative_response(ax,'Yes',behavior_survey.get_ynm(),behavior_survey.get_row_cluster(),behavior_survey.get_col_cluster())
	ax.text(panel_letter_x,panel_letter_y-0.1,'b',size=panel_font_size,weight='bold',transform=ax.transAxes)

	# ax = axes[1,0]
	# ax.axis('off')

	# ax = axes[1,1]
	# ax.axis('off')



	# fig.tight_layout()

	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()


def fig5(behavior_survey,show_plot=False,savefilename=None):
	panel_letter_x = -0.2
	panel_letter_y = 1.05
	panel_font_size = 15

	behavior_survey.cluster_defs()

	# change this to 2,2
	fig,axes = plt.subplots(2,2, figsize=(8,8))

	ax = axes[0,0]
	fields, field_names = behavior_survey.get_academic_fields(compressed=True)
	field_metadata = behavior_survey.get_field_response_rate(fields,'Q57')
	behavior_survey.aesthetics.raw_heatmap(ax,field_metadata,field_names)	
	ax.text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)

	ax = axes[0,1]
	fields, field_names = behavior_survey.get_subfields(compressed=True)
	field_metadata = behavior_survey.get_field_response_rate(fields,'Q57_6_TEXT')
	behavior_survey.aesthetics.raw_heatmap(ax,field_metadata,field_names)	
	ax.text(panel_letter_x,panel_letter_y,'b',size=panel_font_size,weight='bold',transform=ax.transAxes)

	ax = axes[1,0]
	fields, field_names = behavior_survey.get_animals(compressed=True)
	field_metadata = behavior_survey.get_field_response_rate(fields,'Q58')
	behavior_survey.aesthetics.raw_heatmap(ax,field_metadata,field_names)
	ax.text(panel_letter_x,panel_letter_y+0.06,'c',size=panel_font_size,weight='bold',transform=ax.transAxes)

	ax = axes[1,1]
	fields, field_names = behavior_survey.get_seniorities(compressed=False)
	field_metadata = behavior_survey.get_field_response_rate(fields,'Q56')
	behavior_survey.aesthetics.raw_heatmap(ax,field_metadata,field_names)
	ax.text(panel_letter_x,panel_letter_y,'d',size=panel_font_size,weight='bold',transform=ax.transAxes)

	plt.subplots_adjust(wspace=0.5,bottom=0.1)
	# plt.figure.tight_layout()

	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()

def fig6(behavior_survey,show_plot=False,savefilename=None):
	ynm = behavior_survey.get_ynm()
	behavior_survey.cluster_defs()

	behaviorist_df = pd.read_csv('/Users/adamjc/Downloads/behaviorist_cognitivist.csv')
	category = behaviorist_df['category'].to_numpy()
	question_number = behaviorist_df['question_number'].to_numpy()

	# category = category[question_number > 0]
	# question_number = question_number[question_number > 0]
	
	qsort = question_number.argsort()
	category = category[qsort]
	question_number = question_number[qsort]

	num_behaviorist = np.sum(category == 0)
	num_cognitivist = np.sum(category == 1)

	# set xlims and ylims to be the same on either side (e.g., -0.4,+0.4) - or even -1,+1?
	row_cluster = behavior_survey.get_row_cluster()
	palette_metadata = sns.color_palette('hls',len(set(row_cluster)))

	fig = plt.figure(figsize=(8, 8))

	gs = fig.add_gridspec(2,2, width_ratios=(7,2), height_ratios=(2,7), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
	ax = fig.add_subplot(gs[1, 0])
	ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
	ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

	print('the correlation between behaviorism and cognition is:')
	print(pearsonr([np.sum(ans*(category == 0).astype(np.int32)) for ans in ynm]/num_behaviorist,[np.sum(ans*(category == 1).astype(np.int32)) for ans in ynm]/num_behaviorist))
	for rr in set(row_cluster):
		# but really want all the answers so I can make error bars
		behaviorist = [np.sum(ans*(category == 0).astype(np.int32)) for ans in ynm[row_cluster == rr]]/num_behaviorist
		cognitivist = [np.sum(ans*(category == 1).astype(np.int32)) for ans in ynm[row_cluster == rr]]/num_cognitivist

		ax.errorbar(np.mean(behaviorist),np.mean(cognitivist),xerr=np.std(behaviorist)/np.sqrt(len(behaviorist)),yerr=np.std(cognitivist)/np.sqrt(len(cognitivist)),
							fmt='o',color=palette_metadata[rr-1],markeredgecolor='k',ecolor='lightgray',elinewidth=2,capsize=0)

	# make the axes in the center of the plot instead of the outside
	# then can add "cognitivism" and "behaviorism"
	# https://stackoverflow.com/questions/31556446/how-to-draw-axis-in-the-middle-of-the-figure

	ax.legend(['A','B','C','D','E','F','G'],loc='lower left',prop={'size':12},frameon=False)

	# ax.set(xlabel='behaviorist',ylabel='cognitivist',xlim=(-1,1),ylim=(-1,1))
	ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05))
	ax.spines['left'].set_position('center')
	ax.spines['bottom'].set_position('center')
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')


	# plt.show()

	# do I want this as part of the figure above (histograms on X and Y axes?) or as part of supplemental?
	# ax = plt.figure()
	behaviorist = [np.sum(ans*(category == 0).astype(np.int32)) for ans in ynm]/num_behaviorist
	cognitivist = [np.sum(ans*(category == 1).astype(np.int32)) for ans in ynm]/num_cognitivist
	ax_histx.hist(behaviorist,bins=np.arange(-1,1,0.1),histtype='stepfilled',color='grey')
	ax_histx.set(title='behaviorists',ylabel='number respondents')
	ax_histy.hist(cognitivist,bins=np.arange(-1,1,0.1),histtype='stepfilled',color='grey',orientation='horizontal')
	ax_histy.set(title='cognitivists',xlabel='number respondents')

	ax_histx.spines['top'].set_visible(False)
	ax_histx.spines['bottom'].set_visible(False)
	ax_histx.spines['left'].set_visible(False)
	ax_histx.spines['right'].set_visible(False)
	ax_histx.set_xticks([])

	ax_histy.spines['top'].set_visible(False)
	ax_histy.spines['bottom'].set_visible(False)
	ax_histy.spines['left'].set_visible(False)
	ax_histy.spines['right'].set_visible(False)
	ax_histy.set_yticks([])

	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()

def print_cluster_questions(behavior_survey):
	behavior_survey.cluster_defs()
	col_cluster = behavior_survey.get_col_cluster()
	save_df = behavior_survey.get_original_df()

	for cluster in set(col_cluster):
		print('cluster num: ' + str(cluster))
		for elm in np.where(col_cluster == cluster)[0]:
			print(save_df['Q' + str(elm+2)].iloc[0])



def supp_fig1(behavior_survey,show_plot=False,savefilename=None):
	panel_letter_x = -0.2
	panel_letter_y = 1.1
	panel_font_size = 15

	fig,axes = plt.subplots(4,2, figsize=(10,12))
	fig.subplots_adjust(hspace=.5)

	axes[0,0].axis('off')
	ax = axes[0,1]
	agreement_level = np.max(np.array([behavior_survey.get_yes_response(),behavior_survey.get_no_response(),behavior_survey.get_maybe_response()]),axis=0)
	hist_steps = np.arange(0,1,0.01)
	pdf,bins = np.histogram(agreement_level,hist_steps)

	ax.step(bins[:-1],np.cumsum(pdf),'k')
	ax.set(xlabel='highest answer agreement',ylabel='cumulative answers')
	ax.text(panel_letter_x-1.1,panel_letter_y,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)


	axes[1,0].axis('off')
	resp_list = behavior_survey.get_most_answer(behavior_survey.get_yes_response())

	ax = axes[1,1]
	df = pd.DataFrame(resp_list,columns=['Questions','Yes','Maybe','No'])
	df = pd.melt(df,id_vars='Questions',var_name='answers',value_name='response')
	ax = sns.barplot(ax=ax,y='Questions',x='response',hue='answers',data=df,palette=behavior_survey.aesthetics.YMN_palette())
	# sns.despine(ax=ax)

	ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center',size=8)
	ax.set(xlim=(0, 1),title='responses with most Yes',xlabel=None)
	ax.text(panel_letter_x-1.1,panel_letter_y,'b',size=panel_font_size,weight='bold',transform=ax.transAxes)

	axes[2,0].axis('off')
	resp_list = behavior_survey.get_most_answer(behavior_survey.get_no_response())

	ax = axes[2,1]
	df = pd.DataFrame(resp_list,columns=['Questions','Yes','Maybe','No'])
	df = pd.melt(df,id_vars='Questions',var_name='answers',value_name='response')
	ax = sns.barplot(ax=ax,y='Questions',x='response',hue='answers',data=df,palette=behavior_survey.aesthetics.YMN_palette())
	# sns.despine(ax=ax)

	ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center',size=8)
	ax.set(xlim=(0, 1),title='responses with most No',xlabel=None)
	ax.text(panel_letter_x-1.1,panel_letter_y,'c',size=panel_font_size,weight='bold',transform=ax.transAxes)

	axes[3,0].axis('off')
	resp_list = behavior_survey.get_most_answer(behavior_survey.get_maybe_response())

	ax = axes[3,1]
	df = pd.DataFrame(resp_list,columns=['Questions','Yes','Maybe','No'])
	df = pd.melt(df,id_vars='Questions',var_name='answers',value_name='response')
	ax = sns.barplot(ax=ax,y='Questions',x='response',hue='answers',data=df,palette=behavior_survey.aesthetics.YMN_palette())
	# sns.despine(ax=ax)

	ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center',size=8)
	ax.set(xlim=(0, 1),title='responses with most Maybe',xlabel=None)
	ax.text(panel_letter_x-1.1,panel_letter_y,'d',size=panel_font_size,weight='bold',transform=ax.transAxes)


	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()


def supp_fig2(behavior_survey,show_plot=False,savefilename=None):
	panel_letter_x = -0.2
	panel_letter_y = 1.1
	panel_font_size = 15
	dim_1 = 0
	dim_2 = 1

	fig,axes = plt.subplots(2,2, figsize=(8,8))
	fig.subplots_adjust(hspace=.5)

	axes[0,0].axis('off')
	ax = axes[0,1]
	eigs = behavior_survey.get_mca_eigenvalues()
	print(np.cumsum(eigs)/np.sum(eigs))
	ax.plot(eigs/np.sum(eigs),'ko')
	ax.set(xlabel='eigenvalue',ylabel='variance explained')
	ax.text(panel_letter_x-1.1,panel_letter_y,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)


	axes[1,0].axis('off')
	ax = axes[1,1]
	questions = []
	loadings = []
	components_ = behavior_survey.get_mca_components()
	row_names = components_.index.values
	components_ = np.array(components_)
	save_df = behavior_survey.get_original_df()

	loading_srt = components_[:,dim_1].argsort()
	for elm in loading_srt[::-1][:3]:
		questions.append(save_df[row_names[elm].split('_')[0]].iloc[0] + ' ' + row_names[elm].split('_')[1])
		loadings.append(components_[:,dim_1][elm])
	for elm in loading_srt[:3]:
		questions.append(save_df[row_names[elm].split('_')[0]].iloc[0] + ' ' + row_names[elm].split('_')[1])
		loadings.append(components_[:,dim_1][elm])

	ax.hlines(range(0,-len(loadings),-1),0,loadings,colors='b',linewidth=5.0)
	num_dim1_loadings = len(loadings)

	loadings = []
	loading_srt = components_[:,dim_2].argsort()
	for elm in loading_srt[::-1][:3]:
		questions.append(save_df[row_names[elm].split('_')[0]].iloc[0] + ' ' + row_names[elm].split('_')[1])
		loadings.append(components_[:,dim_2][elm])
	for elm in loading_srt[:3]:
		questions.append(save_df[row_names[elm].split('_')[0]].iloc[0] + ' ' + row_names[elm].split('_')[1])
		loadings.append(components_[:,dim_2][elm])

	axes[1,0].axis('off')
	ax = axes[1,1]
	ax.hlines(range(-num_dim1_loadings,-num_dim1_loadings-len(loadings),-1),0,loadings,colors='r',linewidth=5.0)
	# axes[0,0].tick_params(axis='x',labelsize=6)
	ax.set_yticks(range(0,-num_dim1_loadings-len(loadings),-1))
	ax.set_yticklabels(questions,size=6)
	# axes[0,0].set(ylim=(0, 1),title='Largest Factor 1 loadings',xlabel=None)
	ax.set(title='Largest factor loadings',xlabel=None)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
	ax.legend(['Factor ' + str(dim_1),'Factor ' + str(dim_2)],loc='center left',prop={'size':10},bbox_to_anchor=(1,0.85),frameon=False)
	ax.text(panel_letter_x-2.15,panel_letter_y,'b',size=panel_font_size,weight='bold',transform=ax.transAxes)


	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()

def supp_fig3(behavior_survey,show_plot=False,savefilename=None):
	fig,axes = plt.subplots(1,2, figsize=(12,4))
	fig.subplots_adjust(hspace=.5,wspace=.3)

	dim_1=0
	dim_2=1
	panel_letter_x = -0.2
	panel_letter_y = 1.1
	panel_font_size = 15
	bbox_to_anchor = (1,0.5)
	legend_font_size = 8
	box_width_scale = 0.6

	question_dim = np.array(behavior_survey.get_mca_transformation())

	# panel 1: by field	
	ax = axes[0]
	academic_fields, academic_field_names = behavior_survey.get_academic_fields(compressed=False)
	subjects = behavior_survey.get_field_responses(academic_fields,'Q57')
	behavior_survey.aesthetics.plot_dots(ax,question_dim,dim_1,dim_2,academic_fields,subjects)
	behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'Factor ' + str(dim_1),'Factor ' + str(dim_2),academic_field_names,legend_font_size,bbox_to_anchor)
	ax.text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)


	# panel 3: by animal model
	ax = axes[1]
	animals, animal_names = behavior_survey.get_animals(compressed=False)
	subjects = behavior_survey.get_field_responses(animals,'Q58')
	behavior_survey.aesthetics.plot_dots(ax,question_dim,dim_1,dim_2,animals,subjects)
	behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'Factor ' + str(dim_1),'Factor ' + str(dim_2),animal_names,legend_font_size,bbox_to_anchor)
	ax.text(panel_letter_x,panel_letter_y,'b',size=panel_font_size,weight='bold',transform=ax.transAxes)

	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()

def supp_fig4(behavior_survey,show_plot=False,savefilename=None):
	fig,axes = plt.subplots(2,2, figsize=(11,6))
	fig.subplots_adjust(hspace=.5,wspace=.3)

	dim_1=2
	dim_2=3
	panel_letter_x = -0.4
	panel_letter_y = 1.1
	panel_font_size = 15
	bbox_to_anchor = (1,0.5)
	legend_font_size = 8
	box_width_scale = 0.6

	question_dim = np.array(behavior_survey.get_mca_transformation())

	# panel 1: by field	
	ax = axes[0,0]
	academic_fields, academic_field_names = behavior_survey.get_academic_fields(compressed=True)
	subjects = behavior_survey.get_field_responses(academic_fields,'Q57')
	behavior_survey.aesthetics.plot_dots(ax,question_dim,dim_1,dim_2,academic_fields,subjects)
	behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'Factor ' + str(dim_1),'Factor ' + str(dim_2),academic_field_names,legend_font_size,bbox_to_anchor)
	ax.text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)


	# panel 2: by subfield
	ax = axes[0,1]
	sub_fields, sub_field_names = behavior_survey.get_subfields()
	subjects = behavior_survey.get_field_responses(sub_fields,'Q57_6_TEXT')
	behavior_survey.aesthetics.plot_dots(ax,question_dim,dim_1,dim_2,sub_fields,subjects)
	behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'Factor ' + str(dim_1),'Factor ' + str(dim_2),sub_field_names,legend_font_size,bbox_to_anchor)
	ax.text(panel_letter_x,panel_letter_y,'b',size=panel_font_size,weight='bold',transform=ax.transAxes)

	# panel 3: by animal model
	ax = axes[1,0]
	animals, animal_names = behavior_survey.get_animals(compressed=True)
	subjects = behavior_survey.get_field_responses(animals,'Q58')
	behavior_survey.aesthetics.plot_dots(ax,question_dim,dim_1,dim_2,animals,subjects)
	behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'Factor ' + str(dim_1),'Factor ' + str(dim_2),animal_names,legend_font_size,bbox_to_anchor)
	ax.text(panel_letter_x,panel_letter_y,'c',size=panel_font_size,weight='bold',transform=ax.transAxes)

	# panel 4: by seniority
	ax = axes[1,1]
	seniority, seniority_names = behavior_survey.get_seniorities(compressed=False)
	subjects = behavior_survey.get_field_responses(seniority,'Q56')
	behavior_survey.aesthetics.plot_dots(ax,question_dim,dim_1,dim_2,seniority,subjects)
	behavior_survey.aesthetics.clean_dot_plot(ax,box_width_scale,'Factor ' + str(dim_1),'Factor ' + str(dim_2),seniority_names,legend_font_size,bbox_to_anchor)
	ax.text(panel_letter_x,panel_letter_y,'d',size=panel_font_size,weight='bold',transform=ax.transAxes)

	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()

def supp_fig5(behavior_survey,show_plot=False,savefilename=None):
	panel_letter_x = -0.4
	panel_letter_y = 1.1
	panel_font_size = 15

	behavior_survey.cluster_defs()

	fig,axes = plt.subplots(1,2, figsize=(8,5))
	# heatmap
	
	ax = axes[0]
	behavior_survey.aesthetics.plot_cluster_mean_response(ax,'Maybe',behavior_survey.get_ynm(),behavior_survey.get_row_cluster(),behavior_survey.get_col_cluster())
	ax.text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)

	ax = axes[1]
	behavior_survey.aesthetics.plot_cluster_mean_response(ax,'No',behavior_survey.get_ynm(),behavior_survey.get_row_cluster(),behavior_survey.get_col_cluster())
	ax.text(panel_letter_x,panel_letter_y,'b',size=panel_font_size,weight='bold',transform=ax.transAxes)

	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()

def supp_fig6(behavior_survey,show_plot=False,savefilename=None):
	panel_letter_x = -0.4
	panel_letter_y = 1.1
	panel_font_size = 15

	behavior_survey.cluster_defs()

	fig,axes = plt.subplots(2,2, figsize=(8,8))
	# heatmap
	
	ax = axes[0,0]
	ax.axis('off')

	ax = axes[0,1]
	behavior_survey.aesthetics.plot_cluster_relative_response(ax,'Yes',behavior_survey.get_ynm(),behavior_survey.get_row_cluster(),behavior_survey.get_col_cluster())
	ax.text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)

	ax = axes[1,0]
	behavior_survey.aesthetics.plot_cluster_relative_response(ax,'Maybe',behavior_survey.get_ynm(),behavior_survey.get_row_cluster(),behavior_survey.get_col_cluster())
	ax.text(panel_letter_x,panel_letter_y,'b',size=panel_font_size,weight='bold',transform=ax.transAxes)

	ax = axes[1,1]
	behavior_survey.aesthetics.plot_cluster_relative_response(ax,'No',behavior_survey.get_ynm(),behavior_survey.get_row_cluster(),behavior_survey.get_col_cluster())
	ax.text(panel_letter_x,panel_letter_y,'c',size=panel_font_size,weight='bold',transform=ax.transAxes)

	fig.tight_layout()

	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()

def supp_fig7(behavior_survey,show_plot=False,savefilename=None):
	# fix this
	panel_letter_x = -0.2
	panel_letter_y = 1.1
	panel_font_size = 15

	behavior_survey.cluster_defs()

	# fig = plt.figure(figsize=(25,6))
	fig,axes = plt.subplots(2,1, figsize=(25,12))
	# gs = fig.add_gridspec(1,3)
	# ax1 = fig.add_subplot(gs[:2])
	# ax2 = fig.add_subplot(gs[2])

	# fig,axes = plt.subplots(1,3, figsize=(16,6))

	ax = axes[0]
	# ax = ax1
	fields, field_names = behavior_survey.get_academic_fields(compressed=False)
	field_metadata = behavior_survey.get_field_response_rate(fields,'Q57')
	behavior_survey.aesthetics.raw_heatmap(ax,field_metadata,field_names)
	ax.text(panel_letter_x,panel_letter_y,'a',size=panel_font_size,weight='bold',transform=ax.transAxes)

	ax = axes[1]
	# ax = ax2
	fields, field_names = behavior_survey.get_animals(compressed=False)
	field_metadata = behavior_survey.get_field_response_rate(fields,'Q58')
	behavior_survey.aesthetics.raw_heatmap(ax,field_metadata,field_names)
	ax.text(panel_letter_x,panel_letter_y,'b',size=panel_font_size,weight='bold',transform=ax.transAxes)


	if savefilename is not None:
		plt.savefig(savefilename)
	if show_plot:
		plt.show()


def print_metadata_table(behavior_survey,savefilename=None):
	behavior_survey.latex_metadata_table()

def print_tex_table(behavior_survey,savefilename=None):
	behavior_survey.latex_questions_table(behavior_survey.get_original_df())

def make_all_plots(show_plot=True):
	behavior_survey = Survey(survey_file)
	
	fig1(behavior_survey,show_plot=show_plot,savefilename='figs/fig1.pdf')

	fig2(behavior_survey,show_plot=show_plot,savefilename='figs/fig2.pdf')

	fig3(behavior_survey,show_plot=show_plot,savefilename='figs/fig3.pdf')

	fig4(behavior_survey,show_plot=show_plot,savefilename='figs/fig4.pdf')

	# other supplementals to add: factor loadings, eigenvalues, cumulative agreement
	# most agreements
	supp_fig1(behavior_survey,show_plot=show_plot,savefilename='figs/supp_fig1.pdf')
	supp_fig2(behavior_survey,show_plot=show_plot,savefilename='figs/supp_fig2.pdf')

	# tables with questions
	print_tex_table(behavior_survey)

	# https://tex.stackexchange.com/questions/112343/beautiful-table-samples
	# https://twitter.com/neuroecology/status/1335270898681733120
	# https://www.overleaf.com/articles/a-k-roy/httmbqxjpsqp

	# -> should rotate these questions to the original ordering

	# tables with response metadata
	print_metadata_table(behavior_survey)

	supp_fig3(behavior_survey,show_plot=show_plot,savefilename='figs/supp_fig3.pdf')
	supp_fig4(behavior_survey,show_plot=show_plot,savefilename='figs/supp_fig4.pdf')
	supp_fig5(behavior_survey,show_plot=show_plot,savefilename='figs/supp_fig5.pdf')
	supp_fig6(behavior_survey,show_plot=show_plot,savefilename='figs/supp_fig6.pdf')
	supp_fig7(behavior_survey,show_plot=show_plot,savefilename='figs/supp_fig7.pdf')


if __name__ == '__main__':
	behavior_survey = Survey(survey_file)

	make_all_plots(show_plot=False)

	# print_cluster_questions(behavior_survey)

	# manual fixes:
	# Fig2 (clustermap): label on bottom ('questions') is replaced in illustrator because it keeps getting cut off
	# Fig3 (behavior-type definitions): replace illustrations with vectorized versions, move 'example question', add in background
	# shading, add in definition type number (1, 2, etc), replace title text with underlined text, increased distance between 
	# second and third columns
	# Fig 4: create chart entirely in illustrator
	# Fig 5: relabel (cut off), expand everything
	# Fig 6: remove box around legend, remove 'behaviorism', 'cognitivism' dots, label axes (on everything)


	# Supp 6: Fonts are too small

	# Remove boxes around all legends

	# Fig 1d-g: need axis labels. These should be relabeled (start on c-f)
	# Fig 4c: realign boxes, change colors to be like Y/N/M. Definitions -> Behavioral Definitions
	# Supp Fig 1: remove lines around graph (like in Fig 1a).
	# similarly add lines around bars here, or remove them from fig 1a

	# Fig 4: split into two figures, one with the two mean/relative Yes responses, one with the illustration.
	# Finish the caption of fig 4
	# Supp Fig 7: 'machine learning' is covered

	# Ask Priya about definition vs archetype, and whether to switch them.
	# OOOOOOPS need to give out Amazon gift card...
	# Think about where in the brain these would be 'localized', or at least more/less common
	# where do monkeys fit in to all of this?
	# is there something about hierarchy of behavior? more complex behaviors in less mechanistic fields? etc
	# I need to come up with the survey to classify your behavior before we release the preprint

	# TODO analysis:
	# We further asked whether this was true ofspecific groups; we might expect, for instance, that those who are highly trained in behavior or cognitivepsychology might tilt more strongly to one side or another.  [REPORT RESULTS HERE; LOOK ATFACULTY VS GRAD STUDENTS AND THEN COGNITIVE PSYCHOLOGY VS SOME OTHERFIELD??]
	# Fig  1c,  accuracy  X%  +/-  Y  %
	# (DOUBLE-CHECK THIS) number of components to reach ~60% variance explained; alternately when individual components no longer explain even ~5% of the variance each	
	# fix the axes for the MCA dot-plots

