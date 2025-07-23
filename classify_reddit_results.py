#!/usr/bin/env python
# coding: utf-8


'''
Authors: Daniel M. Low
License: See license in github repository
'''

import pickle
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')

pd.set_option("display.max_columns", None)
# pd.options.display.width = 0

# Local
from srl_constructs import constructs_in_order


on_colab = False

if on_colab:
  from google.colab import drive
  project_name = 'project_name'
  drive.mount('/content/drive')
  results_dir = f'/content/drive/MyDrive/datum/{project_name}/data/input/'
  output_dir = f'/content/drive/MyDrive/datum/{project_name}/data/output/'
else:
  input_dir = './data/'
  output_dir = './data/output/'

os.makedirs(output_dir, exist_ok=True)



# =================================================================


# Config
# balance = True # balance training set by downsampling
task = 'classification'
target = 'immiment_risk'
normalize_lexicon = True



if task == 'classification':
	dv = 'suicide_ladder_classification'
	if target == 'suicidal_desire':
		balance_values = ['nonsuicidal','suicidal_desire']
	elif target == 'imminent_risk':
		balance_values = ['suicidal_desire','imminent_risk']
	smallest_value = 'imminent_risk'
	n = 1893

elif task == 'regression':

	# config
	dv = 'suicide_ladder_a'
	balance_values = [1,2,3]
	smallest_value = 3


# =================================================================


def generate_feature_importance_df(trained_model, model_name, feature_names, xgboost_method = 'weight', model_name_in_pipeline = 'estimator', lgbm_method='split'):
	'''
	Function to generate feature importance table for methods that use .coef_ from sklearn
	as well as xgboost models.
	both using sklearn pipelines that go into GridsearchCV, where we need to 
	first access the best_estimator to access, for example, the coefficients.
	
	trained_model: sklearn type model object fit to data
	model_name: str among the ones that appear below
	xgboost_method: str, there are a few options: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.get_score     
	'''
	
	#  Feature importance using coefficients for linear models and gini 
	if model_name in ['SGDRegressor', 'Ridge', 'Lasso', 'LogisticRegression', 'LinearSVC']:
		try:
			coefs = list(trained_model.named_steps['model'].coef_)
		except:
			coefs = list(trained_model.best_estimator_.named_steps[model_name_in_pipeline].coef_)                     # Obtain coefficients from GridSearch
		try:
			coefs= pd.DataFrame(coefs,index = ['Coef.'], columns = feature_names).T # make DF
		except:
			coefs= pd.DataFrame(coefs,index=feature_names, columns = ['Coef.']) # make DF
		coefs['Abs. Coef.'] = coefs['Coef.'].abs()  # add column with absolute values to sort by, both positive and negative values are important. 
		coefs= coefs.sort_values('Abs. Coef.', ascending=False).reset_index() # sort by abs value and reset index to add a feature name column
		coefs= coefs.drop(['Abs. Coef.'], axis=1)   # drop abs value, it's job is done
		coefs.index +=1                             # Importance for publication, start index with 1 , as in 1st, 2nd, 3rd
		coefs= coefs.reset_index()                  # turn index into column
		coefs.columns= ['Importance', 'Feature', 'Coef.'] # Clean column names
		feature_importance = coefs.copy()
		return feature_importance
		
	elif model_name in ['LGBMRegressor', 'LGBMClassifier']:    
		try:
			importance_split = trained_model.named_steps[model_name_in_pipeline].booster_.feature_importance(importance_type='split')
			importance_gain = trained_model.named_steps[model_name_in_pipeline].booster_.feature_importance(importance_type='gain')
			# feature_names = trained_model.named_steps[model_name_in_pipeline].booster_.feature_name()
		except:
			importance_split = trained_model.best_estimator_.named_steps[model_name_in_pipeline].booster_.feature_importance(importance_type='split')
			importance_gain = trained_model.best_estimator_.named_steps[model_name_in_pipeline].booster_.feature_importance(importance_type='gain')
			# feature_names = trained_model.best_estimator_.named_steps[model_name_in_pipeline].booster_.feature_name()
		
		feature_importance = pd.DataFrame({'feature': feature_names, 'split': importance_split, 'gain': importance_gain})
		
		# Sort by gain
		feature_importance = feature_importance.sort_values('gain', ascending=False)
		return feature_importance

		

	elif model_name in ['XGBRegressor', 'XGBClassifier']:
		# WARNING it will not return values for features that weren't used: if feature 3 wasn't used there will not be a f3 in the results        
		try:
			feature_importance = trained_model.named_steps[model_name_in_pipeline].get_booster().get_score(importance_type=xgboost_method )
		except:
			feature_importance = trained_model.best_estimator_.named_steps[model_name_in_pipeline].get_booster().get_score(importance_type=xgboost_method )
		feature_importance_keys = list(feature_importance .keys())
		feature_importance_values = list(feature_importance .values())    
		feature_importance = pd.DataFrame(feature_importance_values,index=feature_importance_keys) # make DF
		feature_importance = feature_importance .sort_values(0, ascending=False)
		feature_importance = feature_importance.reset_index()
	
		feature_importance.index +=1
		feature_importance = feature_importance.reset_index()
		feature_importance
		
		
		feature_importance.columns = ['Importance', 'Feature', xgboost_method.capitalize()]
		
		feature_name_mapping = {}
		for i, feature_name_i in enumerate(feature_names):
			feature_name_mapping[f'f{i}'] = feature_name_i
		
		# Or manually edit here: 
		# feature_name_mapping = {'f0': 'Unnamed: 0', 'f1': 'Adult Mortality', 'f2': 'infant deaths', 'f3': 'percentage expenditure', 'f4': 'Hepatitis B', 'f5': 'Measles ', 'f6': ' BMI ', 'f7': 'under-five deaths ', 'f8': 'Polio', 'f9': 'Diphtheria ', 'f10': ' HIV/AIDS', 'f11': ' thinness  1-19 years', 'f12': ' thinness 5-9 years', 'f13': 'Developing'}
		
		feature_importance['Feature'] = feature_importance['Feature'].map(feature_name_mapping )
	# Todo: add feature_importances_ for sklearn tree based models
	# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#feature-importance-based-on-mean-decrease-in-impurity
	
	
		return feature_importance
	else:
		warnings.warn(f'model not specificied for feature importance: {model_name}')
		return None


# =================================================================


liwc_nonsemantic = ['WC','WPS',
 'BigWords',
 'Dic',
 'Linguistic',
 'function',
 'pronoun',
 'ppron',
 'i',
 'we',
 'you',
 'shehe',
 'they',
 'ipron',
 'det',
 'article',
 'number',
 'prep',
 'auxverb',
 'adverb',
 'conj',
 'negate',
 'verb',
 'adj',
 'quantity',
 'AllPunc',
 'Period',
 'Comma',
 'QMark',
 'Exclam',
 'Apostro',
 'OtherP'
]

liwc_semantic = ['Analytic',
 'Clout',
 'Authentic',
 'Tone', 
 'Drives',
 'affiliation',
 'achieve',
 'power',
 'Cognition',
 'allnone',
 'cogproc',
 'insight',
 'cause',
 'discrep',
 'tentat',
 'certitude',
 'differ',
 'memory',
 'Affect',
 'tone_pos',
 'tone_neg',
 'emotion',
 'emo_pos',
 'emo_neg',
 'emo_anx',
 'emo_anger',
 'emo_sad',
 'swear',
 'Social',
 'socbehav',
 'prosocial',
 'polite',
 'conflict',
 'moral',
 'comm',
 'socrefs',
 'family',
 'friend',
 'female',
 'male',
 'Culture',
 'politic',
 'ethnicity',
 'tech',
 'Lifestyle',
 'leisure',
 'home',
 'work',
 'money',
 'relig',
 'Physical',
 'health',
 'illness',
 'wellness',
 'mental',
 'substances',
 'sexual',
 'food',
 'death',
 'need',
 'want',
 'acquire',
 'lack',
 'fulfill',
 'fatigue',
 'reward',
 'risk',
 'curiosity',
 'allure',
 'Perception',
 'attention',
 'motion',
 'space',
 'visual',
 'auditory',
 'feeling',
 'time',
 'focuspast',
 'focuspresent',
 'focusfuture',
 'Conversation',
 'netspeak',
 'assent',
 'nonflu',
 'filler']







# # Clean up results table
# =================================================================


def insert_empty_row(df, index_to_insert):
	# Splitting the DataFrame
	df_before = df.iloc[:index_to_insert, :]
	df_after = df.iloc[index_to_insert:, :]

	# Creating an empty row (all values set to NaN or any desired value)
	# The length of the empty DataFrame should match the number of columns in the original DataFrame
	empty_row = pd.DataFrame({col: np.nan for col in df.columns}, index=[index_to_insert])

	# Adjusting the index for df_after to accommodate the new row
	df_after.index = df_after.index + 1

	# Concatenating the DataFrames
	df_updated = pd.concat([df_before, empty_row, df_after])

	# Resetting the index if desired
	df_updated = df_updated.reset_index(drop=True)
	return df_updated


# IMPORTANT: Change to correct file
timestamp = '25-03-18T17-25-24'# '25-03-17T19-52-28' 


metrics_to_keep = ['Feature vector', 'Model', 'Sensitivity', 'Specificity', 'Precision',
       'FNR', 'F1', 'ROC AUC', 'PR AUC', 'Best th PR AUC', 'Gridsearch',
       'Best parameters']





prompt_names = {
	'self_harm': 'Self harm',
 'suicide': 'suicide',
 'bully': 'bullying',
 'abuse_sexual': 'sexual abuse',
 'bereavement': 'bereavement',
 'isolated': 'Isolation',
 'anxiety': 'Anxiety',
 'depressed': 'Depression',
 'gender': 'Gender identity',
 'eating': 'Eating disorder',
 'substance': 'Substance use'}


feature_vectors_clean = {
						 
						 "srl_validated": "SRL (50) LogReg",
						 'liwc22':"LIWC-22 (117) LogReg",
						#  'liwc22_semantic':"LIWC-22 only semantic (85)",
						#  "SRL GPT-4 Turbo": "SRL GPT-4 (49)",
						#  "srl_unvalidated": "SRL GPT-4 + manual (49)",
						 
						#  "cts_token_clause": "CTS 1 protoype (1)",
						#  "text_descriptives": "Linguistic (N)",
						#  "SRL GPT-4 Turbo_text_descriptives": "SRL GPT-4 + others (N)",
						#  "srl_unvalidated_text_descriptives": "SRL unvalidated + linguistic (N)",
						#  "srl_validated_text_descriptives": "SRL validated + linguistic (N)",
						#  "all-MiniLM-L6-v2": "all-MiniLM-L6-v2 (384)",
						#  "RoBERTa":'RoBERTa (768)'
						 }


input_dir = './data/output/reddit_binary_classification/'

report_content_13 = False 
sample_sizes = [50,150,2000] 
model_names = ['LogisticRegression']

# Summarized results
all_results = [] # to add across types of feature vectors
all_results_content_validity_3 = []
df_i_full_content_validity_all = []


# for test_set in ['results_content_validity']:
for test_set in ['results', 'results_content_validity']:

	results_all_features_and_n = []
	results_all_features_and_n_all = []
	

	for n in sample_sizes:
		print(n)
		
		for model in model_names:	
			print(model)
			results_df = []
			
			results_dir = f'results_{timestamp}_{n}/'
			
		
			for feature in feature_vectors_clean.keys():
				print(feature)
				
				df = pd.read_csv(f'{input_dir}{results_dir}{test_set}_{n}_{timestamp}.csv')
				
				if test_set == 'results_content_validity':

					# df['prototypicality'] = [1.3, 3]* int(len(df)/2)
					df['prototypicality'] = [3]* int(len(df))
				
				
				df['Classes'] = [eval(n)[-1].replace('_', ' ') for n in df['Classes'].values]
				if test_set == 'results':
					df = df [(df['Class'] == 1) & (df['Average'] == 'binary')] #only these metrics
				
				
				if test_set == 'results_content_validity':
					df_i = df[(df['Feature vector']==feature) & (df['prototypicality'] == 3)]
					
				else:
					df_i = df[df['Feature vector']==feature]
					
				# full results for appendix or repo
				metrics_to_keep_full = ['Feature vector','Classes',  'Sensitivity','Specificity', 'Precision','ROC AUC']

				if test_set == 'results':
					df_i_full = df_i[metrics_to_keep_full]
					df_i_full['Feature vector'] = df_i_full['Feature vector'].replace(feature_vectors_clean)
					df_i_full['N'] = [n]* len(df_i_full)
					# send N column to the second position
					df_i_full = df_i_full[['N'] + [col for col in df_i_full.columns if col != 'N']]
					df_i_full = df_i_full.round(2)
					df_i_full.to_csv(f'{input_dir}{results_dir}{test_set}_{n}_{timestamp}_full_{feature}.csv', index=False)
					
				elif test_set == 'results_content_validity':
					df_i_full_content_validity = pd.read_csv(f'{input_dir}{results_dir}results_{n}_{timestamp}_full_{feature}.csv')
					df_i_full_content_validity['Content validity'] = np.round(df_i['Sensitivity'].values,2)
					df_i_full_content_validity_all.append(df_i_full_content_validity)
					
					
				# summarized results
				if test_set == 'results':
					
					metric_name = ['ROC AUC']
					# metric_name = ['ROC AUC']
				elif test_set == 'results_content_validity':
					metric_name = ['Sensitivity']

				
				df_i = df_i[['Model', 'Classes' ]+metric_name].T
				# df_i.index = ['Construct','Model', df_i.iloc[1,1]]
				df_i.drop(df_i.index[0], inplace=True) # drop Model
				# remove header
				df_i.columns = df_i.iloc[0]
				df_i.drop(df_i.index[0], inplace=True)
				
				
				df_i2 = df_i.copy()	
				df_i3 = df_i.copy()	
				
				df_i2['Summary'] = [f'{np.round(df_i.mean(axis=1).values[i],2)} [{np.round(df_i.min(axis=1).values[i],2)}-{np.round(df_i.max(axis=1).values[i],2)}]' for i in range(len(metric_name))]
				df_i3[metric_name] = [f'{np.round(df_i.mean(axis=1).values[i],2)} [{np.round(df_i.min(axis=1).values[i],2)}-{np.round(df_i.max(axis=1).values[i],2)}]' for i in range(len(metric_name))]

				df_i2['N'] = n*2
				df_i2['Feature'] = f'{feature}'
				df_i2['Model'] = f'{model}'
				df_i3['N'] = n*2
				df_i3['Feature'] = f'{feature}'
				df_i3['Model'] = f'{model}'
				results_all_features_and_n_all.append(df_i2)
				results_all_features_and_n.append(df_i3)
		
	
	
	# Summarized
	results_all_features_and_n = pd.concat(results_all_features_and_n, axis=0)
	results_all_features_and_n.drop(df['Classes'].unique().tolist(),axis=1, inplace=True)
	results_all_features_and_n.reset_index(drop=True, inplace=True)
	results_all_features_and_n.drop_duplicates(keep='last', inplace=True)
	results_all_features_and_n.sort_values(['Model', 'Feature', 'N'], inplace=True)
	# results_all_features_and_n = results_all_features_and_n	
	results_all_features_and_n = results_all_features_and_n[['Feature', 'Model','N']+ metric_name]

	results_all_features_and_n.reset_index(drop=True, inplace=True)
	results_all_features_and_n['Feature'] = [feature_vectors_clean.get(n) for n in results_all_features_and_n['Feature']]
	results_all_features_and_n.drop(['Model'], axis=1, inplace=True)
	
	if test_set == 'results':
		all_results.append(results_all_features_and_n)
	elif test_set == 'results_content_validity':
		all_results_content_validity_3.append(results_all_features_and_n)
	# results_all_features_and_n.to_csv(f'{input_dir}{test_set}_all_features_and_n.csv', index=True)
	results_all_features_and_n_all_constructs = pd.concat(results_all_features_and_n_all, axis=0)


display(all_results[0])
display(all_results_content_validity_3[0])


all_results[0]['Content validity'] = all_results_content_validity_3[0]['Sensitivity']

print(all_results[0][['ROC AUC', 'Content validity']].to_latex())



# cts
# ==========================================================================================

for test_set in ['results', 'results_content_validity']:
	results_all_features_and_n = []
	for feature in ['cts_single', 'cts_multi']:

		if feature == 'cts_single':
			if test_set == 'results':
				path = 'results_25-03-19T20-35-43_cts_single/results_0_25-03-19T20-35-43.csv'
			elif test_set == 'results_content_validity':
				path = 'results_25-03-19T20-35-43_cts_single/results_content_validity_0_25-03-19T20-35-43.csv'
		elif feature == 'cts_multi':
			if test_set == 'results':
				path = 'results_25-03-19T20-35-43_cts_multi/results_0_25-03-19T20-35-43.csv'
			elif test_set == 'results_content_validity':
				path = 'results_25-03-19T20-35-43_cts_multi/results_content_validity_0_25-03-19T20-35-43.csv'
		
		df = pd.read_csv('./data/output/reddit_binary_classification/'+path)
		df['Construct'] = [eval(n)[1] for n in df['Classes'].values]	
		df['Model'] = df['Feature vector']
		if test_set == 'results':
			df = df [(df['Class'] == 1) & (df['Average'] == 'binary')] #only these metrics
	
		df_i = df[(df['Feature vector']==feature)]
		
		if test_set == 'results':
			metric_name = 'ROC AUC'
		elif test_set == 'results_content_validity':
			metric_name = 'Sensitivity'

			
		# WARNING: this only works if you keep this structure:
		# 	for test_set in ['results', 'results_content_validity']:
			# for feature in ['cts_single', 'cts_multi']:
	
		if test_set == 'results':
			metrics_to_keep_full = ['Feature vector',
				'Construct',
				'Sensitivity',
				'Specificity',
				'Precision',
				'ROC AUC']
			df_to_append = df[metrics_to_keep_full]
			df_to_append.rename(columns = {
				'Construct': 'Classes',
				}, inplace = True)
			# add column N = 0, in first position
			df_to_append.insert(0, 'N', 0)
		elif test_set == 'results_content_validity' and feature == 'cts_multi':
			df_to_append['Content validity'] = df['Sensitivity'].round(2).values
			
			df_i_full_content_validity_all.append(df_to_append)
		

		df_i = df_i[['Construct', 'Model', metric_name]].T
		df_i.index = ['Construct','Model', df_i.iloc[1,1]]
		df_i.columns = df_i.iloc[0]
		df_i.drop(df_i.index[1], inplace=True)
		
		df_i.drop(df_i.index[0], inplace=True)
		

		
		df_i = df_i.astype(float).round(2)

	
		df_i[metric_name] = f'{np.round(df_i.mean(axis=1).values[0],2)} [{np.round(df_i.min(axis=1).values[0],2)}-{np.round(df_i.max(axis=1).values[0],2)}]'
		if test_set == 'results_content_validity':
			df_i.rename(columns={'Sensitivity': 'Content validity'}, inplace=True)
		df_i['N'] = 0
		df_i['Prototypicality'] = '3'
		if feature == 'cts_single':
			df_i.index = ['CTS single (1)']
		elif feature == 'cts_multi':
			df_i.index = ['CTS multi (1)']
		
		results_all_features_and_n.append(df_i)
		
	results_all_features_and_n = pd.concat(results_all_features_and_n, axis=0)
	results_all_features_and_n.reset_index(inplace=True)	
	results_all_features_and_n.rename(columns ={'index':'Feature'}, inplace=True)
	
	if test_set == 'results':
		all_results.append(results_all_features_and_n[['Feature','N', 'ROC AUC']])
	elif test_set == 'results_content_validity':
		all_results_content_validity_3.append(results_all_features_and_n[['Feature','N', 'Content validity']])
	



# LLMs
# ==================================================

# Todo content_validity
# for test_set in ['results', 'results_content_validity']:
for test_set in ['results']:
	results_all_features_and_n = []
	for feature in ['google/gemini-2.0-flash-001', 'openai/gpt-4o']:

		if feature == 'google/gemini-2.0-flash-001':
			if test_set == 'results':
				path = 'results_google-gemini-2.0-flash-001_25-04-03T20-59-24.csv'
			# elif test_set == 'results_content_validity':
			# 	path = 'results_25-03-19T20-35-43_cts_single/results_content_validity_0_25-03-19T20-35-43.csv'
		elif feature == 'openai/gpt-4o':
			if test_set == 'results':
				path = 'results_openai-gpt-4o_25-04-03T20-59-24.csv'
			# elif test_set == 'results_content_validity':
			# 	path = 'results_25-03-19T20-35-43_cts_multi/results_content_validity_0_25-03-19T20-35-43.csv'
		
		df = pd.read_csv('./data/output/llms/'+path)
		df['Construct'] = [eval(n)[1] for n in df['Classes'].values]	
		# df['Model'] = df['Feature vector']
		if test_set == 'results':
			df = df [(df['Class'] == 1) & (df['Average'] == 'binary')] #only these metrics
	
		df_i = df[(df['Model']==feature)]
		
		if test_set == 'results':
			metric_name = 'ROC AUC'
		elif test_set == 'results_content_validity':
			metric_name = 'Sensitivity'

			
		# WARNING: this only works if you keep this structure:
		# 	for test_set in ['results', 'results_content_validity']:
			# for feature in ['cts_single', 'cts_multi']:
	
		if test_set == 'results':
			metrics_to_keep_full = ['Feature vector',
				'Construct',
				'Sensitivity',
				'Specificity',
				'Precision',
				'ROC AUC']
			df_to_append = df[metrics_to_keep_full]
			df_to_append.rename(columns = {
				'Construct': 'Classes',
				}, inplace = True)
			# add column N = 0, in first position
			df_to_append.insert(0, 'N', 0)
		elif test_set == 'results_content_validity' and feature == 'cts_multi':
			df_to_append['Content validity'] = df['Sensitivity'].round(2).values
			
			df_i_full_content_validity_all.append(df_to_append)
		

		df_i = df_i[['Construct', 'Model', metric_name]].T
		df_i.index = ['Construct','Model', df_i.iloc[1,1]]
		df_i.columns = df_i.iloc[0]
		df_i.drop(df_i.index[1], inplace=True)
		
		df_i.drop(df_i.index[0], inplace=True)
		

		
		df_i = df_i.astype(float).round(2)

	
		df_i[metric_name] = f'{np.round(df_i.mean(axis=1).values[0],2)} [{np.round(df_i.min(axis=1).values[0],2)}-{np.round(df_i.max(axis=1).values[0],2)}]'
		if test_set == 'results_content_validity':
			df_i.rename(columns={'Sensitivity': 'Content validity'}, inplace=True)
		df_i['N'] = 0
		df_i['Prototypicality'] = '3'
		if feature == 'google/gemini-2.0-flash-001':
			df_i.index = ['Gemini 2.0 Flash']
		elif feature == 'openai/gpt-4o':
			df_i.index = ['GPT-4o']
		
		results_all_features_and_n.append(df_i)
		
	results_all_features_and_n = pd.concat(results_all_features_and_n, axis=0)
	results_all_features_and_n.reset_index(inplace=True)	
	results_all_features_and_n.rename(columns ={'index':'Feature'}, inplace=True)
	
	if test_set == 'results':
		all_results.append(results_all_features_and_n[['Feature','N', 'ROC AUC']])
	elif test_set == 'results_content_validity':
		all_results_content_validity_3.append(results_all_features_and_n[['Feature','N', 'Content validity']])
	

# Summarize all models
# ==================================================

pd.set_option('display.float_format', '{:.2f}'.format)
results_all_df = pd.concat(all_results, axis=0)
results_all_df_content_validity = pd.concat(all_results_content_validity_3, axis=0)

# TODO move this to the bottom after gemma and LLM models 
df_i_full_content_validity_all = pd.concat(df_i_full_content_validity_all, axis=0)
results_dir_clean = results_dir.replace('_2000/', '')
df_i_full_content_validity_all.to_csv(f'./data/output/reddit_binary_classification/{test_set}_{timestamp}_full_content_validity_all.csv', index=False)

display(df_i_full_content_validity_all)




