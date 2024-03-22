#!/usr/bin/env python
# coding: utf-8

# =================================================================


#!pip install -q pyarrow



# =================================================================


'''
Authors: Daniel M. Low
License: See license in github repository
'''

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')

pd.set_option("display.max_columns", None)
# pd.options.display.width = 0


# os.chdir(os.path.dirname(__file__)) # Set working directory to current file

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
# target = 'immiment_risk'
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


# # Skip loading data and extracting featues and load below

# # Or load data and extract

# # Load everything above

# =================================================================


import pickle
run_this = False #True saves, False loads
if run_this:
    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'wb') as f:
        pickle.dump(dfs, f) 
else:

    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'rb') as f:
    	dfs = pickle.load(f)


# =================================================================


from srl_constructs import constructs_in_order




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


# =================================================================



input_dir = './data/output/binary_classification/'



metrics_to_keep = ['Feature vector', 'Model', 'Sensitivity', 'Specificity', 'Precision',
       'FNR', 'F1', 'ROC AUC', 'PR AUC', 'Best th PR AUC', 'Gridsearch',
       'Best parameters']


prompt_names = {
	'self_harm': 'Self harm',
 'suicide': 'suicide',
 'bully': 'bullying',
 'abuse_physical': 'physical abuse',
 'abuse_sexual': 'sexual abuse',
 'relationship': 'relationship issues',
 'bereavement': 'bereavement',
 'isolated': 'Isolation',
 'anxiety': 'Anxiety',
 'depressed': 'Depression',
 'gender': 'Gender identity',
 'eating': 'Eating disorder',
 'substance': 'Substance use'}





feature_vectors_clean = {
						#  'liwc22_semantic':"LIWC-22 only semantic (85)",
						 'liwc22':"LIWC-22 (117)",
						#  "SRL GPT-4 Turbo": "SRL GPT-4 (49)",
						#  "srl_unvalidated": "SRL GPT-4 + manual (49)",
						 "srl_validated": "SRL (50)",
						#  "cts_token_clause": "CTS 1 protoype (1)",
						#  "text_descriptives": "Linguistic (N)",
						#  "SRL GPT-4 Turbo_text_descriptives": "SRL GPT-4 + others (N)",
						#  "srl_unvalidated_text_descriptives": "SRL unvalidated + linguistic (N)",
						#  "srl_validated_text_descriptives": "SRL validated + linguistic (N)",
						#  "all-MiniLM-L6-v2": "all-MiniLM-L6-v2 (384)",
						#  "RoBERTa":'RoBERTa (768)'
						 }


report_content_13 = False 
sample_sizes = [50,150,2000] # TODO
model_names = ['LogisticRegression']
timestamp = '24-03-06T06-17-02'
all_results = []
all_results_content_validity_3 = []

for test_set in ['results', 'results_content_validity']:

	results_all_features_and_n = []

	for n in sample_sizes:
		print(n)
		
		for model in model_names:	
			print(model)
			results_df = []
			
			results_dir = f'results_{timestamp}_{n}/'
			
			

			
			for feature in feature_vectors_clean.keys():
				
				df = pd.read_csv(f'{input_dir}{results_dir}{test_set}_{n}_{timestamp}.csv')
				
				df['Construct'] = [n.split(f'{model}_')[-1] for n in df['Model'].values]
				df['Model'] = [n.split('_')[0] for n in df['Model'].values]
				df['Model'] = df['Feature vector'] + ' ' + df['Model']
				
				df_i = df[df['Feature vector']==feature]
				if test_set == 'results':
					metric_name = 'ROC AUC'
				elif test_set == 'results_content_validity':
					metric_name = 'Sensitivity'

				df_i = df_i[['Construct', 'Model', metric_name]].T
				df_i.index = ['Construct','Model', df_i.iloc[1,1]]
				df_i.drop(df_i.index[1], inplace=True)
				# remove header
				df_i.columns = df_i.iloc[0]
				df_i.drop(df_i.index[0], inplace=True)
				
				# Add column with Mean [min-max]
				# df_i.index.set_index('Construct')
				df_i = df_i.astype(float).round(2)

				if test_set == 'results_content_validity':
					df_i_3 = df_i[[n for n in df_i.columns if n.endswith('-3') ]]
					df_i_3.columns = [n.replace('_content-validity-3', '') for n in df_i_3.columns]
					df_i_13 = df_i[[n for n in df_i.columns if n.endswith('-13') ]]
					df_i_13.columns = [n.replace('_content-validity-13', '') for n in df_i_13.columns]
					
					if report_content_13:
						for df_i, threshold in zip([df_i_3, df_i_13], ['3', '1.3']):
							df_i[metric_name] = f'{np.round(df_i.mean(axis=1).values[0],2)} [{np.round(df_i.min(axis=1).values[0],2)}-{np.round(df_i.max(axis=1).values[0],2)}]'
							df_i['N'] = n*2
							df_i['Prototypicality'] = threshold
							
							results_all_features_and_n.append(df_i)
					else:
						for df_i, threshold in zip([df_i_3], ['3']):
							df_i[metric_name] = f'{np.round(df_i.mean(axis=1).values[0],2)} [{np.round(df_i.min(axis=1).values[0],2)}-{np.round(df_i.max(axis=1).values[0],2)}]'
							df_i['N'] = n*2
							df_i['Prototypicality'] = threshold
							
							results_all_features_and_n.append(df_i)	

					
				else:

					df_i[metric_name] = f'{np.round(df_i.mean(axis=1).values[0],2)} [{np.round(df_i.min(axis=1).values[0],2)}-{np.round(df_i.max(axis=1).values[0],2)}]'
					df_i['N'] = n*2
					results_all_features_and_n.append(df_i)
	results_all_features_and_n = pd.concat(results_all_features_and_n, axis=0)

	# sort index
	results_all_features_and_n.sort_index(inplace=True)
	# results_all_features_and_n.sort_values(['Construct', 'N'], inplace=True)
	results_all_features_and_n.columns.name = 'Model'
	results_all_features_and_n.columns = [prompt_names.get(n).capitalize() if n in prompt_names.keys() else n for n in results_all_features_and_n.columns]
	results_all_features_and_n.index = [n.replace('liwc22', feature_vectors_clean.get('liwc22')).replace('srl_validated', feature_vectors_clean.get('srl_validated')).replace('LogisticRegression', 'LogReg') for n in results_all_features_and_n.index]
	if test_set == 'results':
		all_results.append(results_all_features_and_n)
	elif test_set == 'results_content_validity':
		all_results_content_validity_3.append(results_all_features_and_n)
	# results_all_features_and_n.to_csv(f'{input_dir}{test_set}_all_features_and_n.csv', index=True)


# cts
# ==========================================================================================
feature = 'cts_token_clause'	
test_set = 'results'

file = 'results_50_24-03-07T19-34-40.csv'

for feature in ['cts_token_clause', 'cts_prototypes_clause']:

	for test_set in ['results', 'results_content_validity']:

		if feature == 'cts_token_clause':
			if test_set == 'results':
				path = 'results_24-03-07T19-34-40_cts_token_clause/results_50_24-03-07T19-34-40.csv'
			elif test_set == 'results_content_validity':
				path = 'results_24-03-07T19-34-40_cts_token_clause/results_content_validity_50_24-03-07T19-34-40.csv'
		elif feature == 'cts_prototypes_clause':
			if test_set == 'results':
				path = 'results_24-03-07T23-57-26_cts_prototypes_clause/results_0_24-03-07T23-57-26.csv'
			elif test_set == 'results_content_validity':
				path = 'results_24-03-07T23-57-26_cts_prototypes_clause/results_content_validity_0_24-03-07T23-57-26.csv'
		
		
		results_all_features_and_n = []

		df = pd.read_csv('./data/output/binary_classification/'+path)
		df['Construct'] = [n.split(f'{model}_')[-1] for n in df['Model'].values]
		df['Model'] = [n.split('_')[0] for n in df['Model'].values]
		df['Model'] = df['Feature vector'] + ' ' + df['Model']

		df_i = df[df['Feature vector']==feature]
		if test_set == 'results':
			metric_name = 'ROC AUC'
		elif test_set == 'results_content_validity':
			metric_name = 'Sensitivity'

		df_i = df_i[['Construct', 'Model', metric_name]].T
		df_i.index = ['Construct','Model', df_i.iloc[1,1]]
		df_i.drop(df_i.index[1], inplace=True)
		# remove header
		df_i.columns = [n.split(feature+'_')[-1].split('_thesh')[0] for n in df_i.iloc[0]]
		df_i.drop(df_i.index[0], inplace=True)

		# Add column with Mean [min-max]
		# df_i.index.set_index('Construct')
		df_i = df_i.astype(float).round(2)

		if test_set == 'results_content_validity':
			
			df_i_3 = df_i[[n for n in df_i.columns if n.endswith('-3') ]]
			df_i_3.columns = [n.replace('_content-validity-3', '') for n in df_i_3.columns]
			# df_i_13 = df_i[[n for n in df_i.columns if n.endswith('-13') ]]
			# df_i_13.columns = [n.replace('_content-validity-13', '') for n in df_i_13.columns]
			# for df_i, threshold in zip([df_i_3, df_i_13], ['3', '1.3']):
			
			for df_i, threshold in zip([df_i_3], ['3']):
				df_i[metric_name] = f'{np.round(df_i.mean(axis=1).values[0],2)} [{np.round(df_i.min(axis=1).values[0],2)}-{np.round(df_i.max(axis=1).values[0],2)}]'
				df_i['N'] = n*2
				df_i['Prototypicality'] = threshold
				df_i['N'] = 0
				if feature == 'cts_token_clause':
					df_i.index = ['CTS 1 prototype (1)']
				elif feature == 'cts_prototypes_clause':
					df_i.index = ['CTS many prototypes (1)']

				results_all_features_and_n.append(df_i)
			
		else:

			df_i[metric_name] = f'{np.round(df_i.mean(axis=1).values[0],2)} [{np.round(df_i.min(axis=1).values[0],2)}-{np.round(df_i.max(axis=1).values[0],2)}]'
			df_i['N'] = 0
			if feature == 'cts_token_clause':
				df_i.index = ['CTS 1 prototype (1)']
			elif feature == 'cts_prototypes_clause':
				df_i.index = ['CTS many prototypes (1)']
			results_all_features_and_n.append(df_i)
		results_all_features_and_n = pd.concat(results_all_features_and_n, axis=0)

		# sort index
		results_all_features_and_n.sort_index(inplace=True)
		# results_all_features_and_n.sort_values(['Construct', 'N'], inplace=True)
		results_all_features_and_n.columns.name = 'Model'
		results_all_features_and_n.columns = [prompt_names.get(n).capitalize() if n in prompt_names.keys() else n for n in results_all_features_and_n.columns]
		results_all_features_and_n.index = [n.replace('liwc22', feature_vectors_clean.get('liwc22')).replace('srl_validated', feature_vectors_clean.get('srl_validated')).replace('LogisticRegression', 'LogReg') for n in results_all_features_and_n.index]
		if test_set == 'results':
			all_results.append(results_all_features_and_n)
		elif test_set == 'results_content_validity':
			all_results_content_validity_3.append(results_all_features_and_n)
		# results_all_features_and_n.to_csv(f'{input_dir}{test_set}_all_features_and_n.csv', index=True)



	results_all_df = pd.concat(all_results, axis=0)
	results_all_df = results_all_df[['N', 'ROC AUC', 'Self harm', 'Suicide', 'Bullying', 'Physical abuse', 'Sexual abuse',
		'Relationship issues', 'Bereavement', 'Isolation', 'Anxiety','Depression', 'Gender identity', 'Eating disorder', 'Substance use']]


	# TODO: Add content validity
	results_all_df_content_validity = pd.concat(all_results_content_validity_3, axis=0)
	results_all_df_content_validity = results_all_df_content_validity[['N', 'Sensitivity', 'Self harm', 'Suicide', 'Bullying', 'Physical abuse', 'Sexual abuse',
		'Relationship issues', 'Bereavement', 'Isolation', 'Anxiety','Depression', 'Gender identity', 'Eating disorder', 'Substance use']]

# Gemma 
# ========================================================================

ctl_tags13 = ['self_harm',
 'suicide',
 'bully',
 'abuse_physical',
 'abuse_sexual',
 'relationship',
 'bereavement',
 'isolated',
 'anxiety',
 'depressed',
 'gender',
 'eating',
 'substance']

features = ['gemma-2b-it', 'gemma-7b-it']
feature = 'gemma-2b-it'



gemma_2b_construct_name_mapping = dict(zip(['Self harm', 'Suicide', 'Bully', 'Abuse physical', 'Abuse sexual',
'Relationship', 'Bereavement', 'Isolated', 'Anxiety', 'Depressed',
'Gender', 'Eating', 'Substance'], prompt_names.values()))


for feature in features:	
	for test_set in ['results', 'results_content_validity']: 
		if feature == 'gemma-2b-it':
			if test_set == 'results':
				results_dir = 'google-gemma-2b-it_2024-03-01T16-56-22_600/'
				timestamp = results_dir.split('_')[-2]
				
				path = 'google-gemma-2b-it_2024-03-01T16-56-22_600/results_all_constructs_google-gemma-2b-it_2024-03-01T16-56-22_600.csv'
			elif test_set == 'results_content_validity':
				results_dir = 'google-gemma-2b-it_2024-03-06T21-00-22_construct-validity-3_prototypical-of/'
				timestamp = results_dir.split('_')[1]
				
		elif feature == 'gemma-7b-it':
			if test_set == 'results':

				results_dir = 'google-gemma-7b-it_2024-03-01T02-03-13_600/'
				timestamp = results_dir.split('_')[-2]
			elif test_set == 'results_content_validity':
				results_dir = 'google-gemma-7b-it_2024-03-06T23-29-37_construct-validity-3_prototypical_of/'
				timestamp = results_dir.split('_')[1]
		
		
		results_all_features_and_n = []
		df = []
		

		for dv in ctl_tags13:
			os.listdir('./data/output/binary_classification/'+results_dir+f'{dv}/')
			if test_set == 'results':
			
				if feature == 'gemma-2b-it':
					df_i = pd.read_csv('./data/output/binary_classification/'+results_dir+f'{dv}/results_google-{feature}_{timestamp}.csv', index_col=0)
				elif feature == 'gemma-7b-it':
					df_i = pd.read_csv('./data/output/binary_classification/'+results_dir+f'{dv}/results_google-{feature}_{dv}_{timestamp}.csv', index_col=0)
			elif test_set == 'results_content_validity':
				files = os.listdir('./data/output/binary_classification/'+results_dir+f'{dv}/')
				file = [n for n in files if 'results' in n][0]
				
				df_i = pd.read_csv('./data/output/binary_classification/'+results_dir+f'{dv}/{file}', index_col=0)
			

			df.append(df_i)
		df = pd.concat(df, axis=0)
		if test_set == 'results':
			if feature == 'gemma-2b-it':
				df['Construct'] = [gemma_2b_construct_name_mapping.get(n).capitalize() for n in df['Construct'].values]

			if feature == 'gemma-7b-it':
				df['Construct'] = [n.split(f'{model}_')[-1].split(feature+'_')[-1] for n in df['Model'].values]
		elif test_set == 'results_content_validity':
			df['Construct'] = [n.split(f'{model}_')[-1].split(feature+'_')[-1] for n in df['Model'].values]

		# df['Model'] = [n.split('_')[0] for n in df['Model'].values]
		# df['Model'] = df['Feature vector'] + ' ' + df['Model']

		# df_i = df[df['Feature vector']==feature]
		if test_set == 'results':
			metric_name = 'ROC AUC'
		elif test_set == 'results_content_validity':
			metric_name = 'Sensitivity'

		df_i = df[['Construct', 'Model', metric_name]].T
		df_i.index = ['Construct','Model', df_i.iloc[1,1]]
		df_i.drop(df_i.index[1], inplace=True)
		# remove header
		df_i.columns = [n.split(feature+'_')[-1].split('_thesh')[0] for n in df_i.iloc[0]]
		df_i.drop(df_i.index[0], inplace=True)

		# Add column with Mean [min-max]
		# df_i.index.set_index('Construct')
		df_i = df_i.astype(float).round(2)

		if test_set == 'results_content_validity':
			
			df_i_3 = df_i[[n for n in df_i.columns if n.endswith('-3') ]]
			df_i_3.columns = [n.replace('_content-validity-3', '') for n in df_i_3.columns]
			# df_i_13 = df_i[[n for n in df_i.columns if n.endswith('-13') ]]
			# df_i_13.columns = [n.replace('_content-validity-13', '') for n in df_i_13.columns]
			# for df_i, threshold in zip([df_i_3, df_i_13], ['3', '1.3']):
			for df_i, threshold in zip([df_i_3], ['3']):
				df_i[metric_name] = f'{np.round(df_i.mean(axis=1).values[0],2)} [{np.round(df_i.min(axis=1).values[0],2)}-{np.round(df_i.max(axis=1).values[0],2)}]'
				
				df_i['Prototypicality'] = threshold
				df_i['N'] = 0
				df_i.index = [feature]

				results_all_features_and_n.append(df_i)
			
		else:

			df_i[metric_name] = f'{np.round(df_i.mean(axis=1).values[0],2)} [{np.round(df_i.min(axis=1).values[0],2)}-{np.round(df_i.max(axis=1).values[0],2)}]'
			df_i['N'] = 0
			
			df_i.index = [feature]
			
			results_all_features_and_n.append(df_i)
		results_all_features_and_n = pd.concat(results_all_features_and_n, axis=0)

		# sort index
		results_all_features_and_n.sort_index(inplace=True)
		# results_all_features_and_n.sort_values(['Construct', 'N'], inplace=True)
		results_all_features_and_n.columns.name = 'Model'
		results_all_features_and_n.columns = [prompt_names.get(n).capitalize() if n in prompt_names.keys() else n for n in results_all_features_and_n.columns]
		results_all_features_and_n.index = [n.replace('liwc22', feature_vectors_clean.get('liwc22')).replace('srl_validated', feature_vectors_clean.get('srl_validated')).replace('LogisticRegression', 'LogReg') for n in results_all_features_and_n.index]
		if test_set == 'results':
			all_results.append(results_all_features_and_n)
		elif test_set == 'results_content_validity':
			all_results_content_validity_3.append(results_all_features_and_n)
		# results_all_features_and_n.to_csv(f'{input_dir}{test_set}_all_features_and_n.csv', index=True)



	results_all_df = pd.concat(all_results, axis=0)
	results_all_df = results_all_df[['N', 'ROC AUC', 'Self harm', 'Suicide', 'Bullying', 'Physical abuse', 'Sexual abuse',
		'Relationship issues', 'Bereavement', 'Isolation', 'Anxiety','Depression', 'Gender identity', 'Eating disorder', 'Substance use']]


	# TODO: Add content validity
	results_all_df_content_validity = pd.concat(all_results_content_validity_3, axis=0)
	results_all_df_content_validity = results_all_df_content_validity[['N', 'Sensitivity', 'Self harm', 'Suicide', 'Bullying', 'Physical abuse', 'Sexual abuse',
		'Relationship issues', 'Bereavement', 'Isolation', 'Anxiety','Depression', 'Gender identity', 'Eating disorder', 'Substance use']]


results_all_df.to_csv(output_dir+'tables/binary_classification_results.csv', index=True)

results_all_df_content_validity.index = [n.replace(' srl', '') for n in results_all_df_content_validity.index]
results_all_df_content_validity.to_csv(output_dir+'tables/content_validity.csv', index=True)



# =================================================================
import seaborn as sns

feature_vectors = ['liwc22_semantic', 'srl_unvalidated','all-MiniLM-L6-v2']
timestamp = '24-02-16T06-25-10'

model_name = 'LGBMRegressor'

toy = False

for plot_type in ['strip']:

	for feature_vector in feature_vectors:	
		print(model)
		results_df = []
		
		results_dir = f'results_{timestamp}_{n}_regression_{balance_values[-1]}/'
		
		files = os.listdir('./data/output/ml_performance/'+results_dir)
		

		print(model)
		file = [n for n in files if  f"y_pred_{feature_vector}_{model_name}" in n and 'csv' in n]
		if file != []:
			if len(file)==1:	
				y_pred = pd.read_csv('./data/output/ml_performance/'+results_dir+file[0])




		if feature_vector == 'liwc22_semantic':
			y_test = dfs['test']['liwc22_y']
		else:
			X_train, y_train, X_test, y_test = get_splits(feature_vector)


		y_df = y_pred.copy()
		i = 2
		y_df['y_test'] = y_test
		y_df.columns = ['Predictions', 'True scores']

		if toy:
			y_df = y_df.sample(frac=0.20)


		# colorblind friendly https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000
			
		

			
		colors_severity = {
			
			1: '#FFB000',
			2: '#FE6100',
			3: '#DC267F',
			# 1: '#FFBB78',
			# 2: '#FF7F0E',
			# 3: '#D62728' 
			
		}

		# Create a boxplot with the specified color palette
		
		toy
		figsize = (3.25,8)
		plt.figure(figsize=figsize)  # Width=10 inches, Height=6 inches
		# sns.scatterplot(data = y_df, y = 'y_pred',x = 'y_test', alpha = 0.1)


		sns.boxplot(y='Predictions', x='True scores', data=y_df, palette=colors_severity, showfliers=False,
			  boxprops=dict(alpha=1))

		if plot_type == 'swarm':
			sns.swarmplot(y='Predictions', x='True scores', data=y_df,  color='0.25', alpha=0.3)
		elif plot_type == 'strip':
			sns.stripplot(y='Predictions', x='True scores', data=y_df,  color='#648FFF', alpha=0.2,jitter=0.2)

		plt.xticks(ticks = [0,1,2],labels = ['Nonsuicidal', 'Suicidal', 'Imminent risk'])
		plt.ylim((0.4,3.5))


		# Show the plot
		
		plt.tight_layout()
		plt.savefig(f'./data/output/figures/{plot_type}_boxplot_{feature_vector}_{model_name}.png', bbox_inches='tight', dpi=300)
		plt.show()


		


# =================================================================


feature_vector


# # Feature importance plot

# =================================================================


import pickle
with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'rb') as f:
	dfs = pickle.load(f)


# =================================================================


model = 'LGBMRegressor'
n = 'all'


results_dir = f'./data/output/ml_performance/results_{timestamp}_{n}_regression_{balance_values[-1]}/'

files = os.listdir(results_dir)
feature_vectors = ['srl_unvalidated', 'liwc22_semantic']
table_names = ['SRL GPT-4 + manual', 'LIWC-22 semantic']

rank_col_name = 'Rank'
files
feature_importance = []
for file, table_name in zip(feature_vectors, table_names):
    file1 = [n for n in files if ('feature_importance_'+file in n and 'clean' not in n)][0]
    
print(file1)
fi = pd.read_csv(results_dir+file1)
# fi.columns = ['Feature', 'Split', 'Gain']
# fi=fi.drop('Split', axis=1).round(1)
# fi = fi.reset_index()
# fi.columns = [rank_col_name, 'Feature', 'Gain']
# fi[rank_col_name]+=1
# fi[rank_col_name] = fi[rank_col_name].astype(str)
fi


# =================================================================


# For each feature, correlate feature with y
from scipy.stats import spearmanr
import math
liwc22_X = dfs['train']['liwc22_X']
liwc22_y = dfs['train']['liwc22_y']
liwc_rho = {}
for feature in liwc22_X.columns:
	filtered_list1, filtered_list2 = zip(*[(x, y) for x, y in zip(liwc22_y, liwc22_X[feature].values) if not math.isnan(x) and not math.isnan(y)])

	# Converting the tuples back to lists
	filtered_list1 = list(filtered_list1)
	filtered_list2 = list(filtered_list2)
	r,p = spearmanr(filtered_list1, filtered_list2)
	# r,p = spearmanr(liwc22_y, liwc22_X[feature])
	# if p <= 0.05:
	liwc_rho[feature] = np.round(r,2)
	if str(r)=='nan':
		
		print(feature)
	# else:
		# liwc_rho[feature] = np.nan


# For each feature, correlate feature with y
srl_unv_X = dfs['train']['srl_unvalidated']
srl_unv_y = dfs['train']['y']
srl_unv_rho = {}
for feature in srl_unv_X.columns:
	# remove nans:
	filtered_list1, filtered_list2 = zip(*[(x, y) for x, y in zip(srl_unv_y, srl_unv_X[feature].values) if not math.isnan(x) and not math.isnan(y)])

	# Converting the tuples back to lists
	filtered_list1 = list(filtered_list1)
	filtered_list2 = list(filtered_list2)
	r,p = spearmanr(filtered_list1, filtered_list2)
	# if p <= 0.05:
	srl_unv_rho[feature] = np.round(r,2)
	# else:
		# srl_unv_rho[feature] = np.nan
	




# =================================================================


results_dir


# =================================================================


model = 'LGBMRegressor'
files = os.listdir(results_dir)
feature_vectors = ['srl_unvalidated', 'liwc22_semantic']
table_names = ['SRL unvalidated', 'LIWC-22 semantic']

rank_col_name = 'Rank'
files
feature_importance = []
for file, table_name in zip(feature_vectors, table_names):
	# timestamp_i = timestamp.replace('results_', '')
	# if file == 'liwc22_semantic':
	# 	file1 = f'feature_importance_{file}_{model}_gridsearch-True_all_24-02-16T00-03-59.csv'
	# else:
	file1 = f'feature_importance_{file}_{model}_gridsearch-True_all_{timestamp}.csv'
	
	
	
	fi = pd.read_csv(results_dir+file1)
	fi.columns = ['Feature', 'Split', 'Gain']
	fi=fi.drop('Split', axis=1).round(1)
	fi = fi.reset_index()
	fi.columns = [rank_col_name, 'Feature', 'Gain']
	fi[rank_col_name]+=1
	fi[rank_col_name] = fi[rank_col_name].astype(str)
	if 'liwc22' in file:
		fi['rho'] = fi['Feature'].map(liwc_rho)
	else:
		fi['rho'] = fi['Feature'].map(srl_unv_rho)

	fi.to_csv(results_dir+'feature_importance_'+file+'_clean.csv', index=False)
	columns = pd.MultiIndex.from_tuples([
	(table_name, rank_col_name),
	(table_name, 'Feature'),
	(table_name, 'Gain'),
	(table_name, 'rho'),
	])
	fi.columns = columns
	feature_importance.append(fi)

feature_importance_df = pd.concat([feature_importance[0],feature_importance[1].drop(columns=(table_names[1], rank_col_name))],axis=1)



feature_vectors = '_'.join(feature_vectors)



feature_importance_df.to_csv(results_dir+f'feature_importance_{feature_vectors}_gridsearch-True_all_{timestamp}_all.csv', index= 0 )
feature_importance_df.to_csv('./data/output/tables/'+f'feature_importance_{feature_vectors}_gridsearch-True_all_{timestamp}_all.csv', index= 0 )

display(feature_importance_df)

feature_importance_df.iloc[:20].to_csv(results_dir+f'feature_importance_{feature_vectors}_gridsearch-True_all_{timestamp}_top20.csv', index= 0 )

# top 15 and bottom 10
df0 = feature_importance[0].copy()
top_15 = df0.head(15)
bottom_10 = df0.tail(10)
empty_row = pd.DataFrame(np.nan, index=[0], columns=bottom_10.columns)
bottom_10 = pd.concat([empty_row, bottom_10]).reset_index(drop=True)
df0 = pd.concat([top_15, bottom_10])
df0 = df0.reset_index(drop=True)

df1 = feature_importance[1].copy()
top_15 = df1.head(15)
bottom_10 = df1.tail(10)
empty_row = pd.DataFrame(np.nan, index=[0], columns=bottom_10.columns)
bottom_10 = pd.concat([empty_row, bottom_10]).reset_index(drop=True)
df1 = pd.concat([top_15, bottom_10])
df1 = df1.reset_index(drop=True)


feature_importance_df = pd.concat([df0,df1],axis=1)
feature_importance_df.to_csv('./data/output/tables/'+f'feature_importance_{feature_vectors}_gridsearch-True_all_{timestamp}_top_and_bottom.csv', index= 0 )
display(feature_importance_df)



# =================================================================





# =================================================================





# =================================================================





# =================================================================





# =================================================================





# =================================================================





# # Error analysis
# 

# =================================================================





# =================================================================


ts_i = '24-02-15T20-17-48'
n = 'all'

output_dir_i = output_dir + f'results_{ts_i}_{n}_{task}_{balance_values[-1]}/'

results = []
# for gridsearch in [True]:

# for feature_vector in ['srl_unvalidated', 'all-MiniLM-L6-v2']:#['srl_unvalidated']:#, 'srl_unvalidated']:
for feature_vector in feature_vectors:#['srl_unvalidated']:#, 'srl_unvalidated']:
	if feature_vector == 'liwc22_semantic':
		X_train, y_train,X_val, y_val, X_test, y_test = get_splits('liwc22')
		X_train = X_train[liwc_semantic]
		X_val = X_val[liwc_semantic]
		X_test = X_test[liwc_semantic]

	else:
		X_train, y_train,X_val, y_val, X_test, y_test = get_splits(feature_vector)

	


	if toy:
		X_train['y'] = y_train
		X_train = X_train.sample(n = 100)
		y_train = X_train['y'].values
		X_train = X_train.drop('y', axis=1)

	elif n!='all':
		X_train['y'] = y_train
		X_train = X_train.sample(n = n, random_state=42)
		y_train = X_train['y'].values
		X_train = X_train.drop('y', axis=1)


	if task == 'classification':
		encoder = LabelEncoder()

		# Fit and transform the labels to integers
		y_train = encoder.fit_transform(y_train)
		y_test = encoder.transform(y_test)

	
	for model_name in model_names: 
		y_pred = pd.read_csv(output_dir_i+f'y_pred_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}.csv')
		break
	break


# =================================================================


from sklearn import metrics
y_df = y_pred.copy()
i = 2
y_df['y_test'] = y_test
y_df.columns = ['y_pred', 'y_test']
y_df_i = y_df[y_df['y_test']==i]
y_df_i['error'] = y_df_i['y_pred'] - y_df_i['y_test']
y_df_i = y_df_i.sort_values(by='error')
X_test_text = dfs['test']['df_text']
print(X_test_text.shape, y_df.shape)
# display(X_test_text.head(), y_df[:5])
display(y_df_i.iloc[:10])
display(X_test_text.loc[y_df_i.index[:10]])
docs = X_test_text.loc[y_df_i.index[:10]]['text'].to_list()

print(docs)
# metrics.mean_absolute_error(y_test, y_pred.values)


# =================================================================


import dill
sys.path.append( './../../concept-tracker/') # TODO: replace with pip install construct-tracker
from concept_tracker import lexicon


def load_lexicon(path):
	lexicon = dill.load(open(path, "rb"))
	return lexicon
srl = load_lexicon("./data/input/lexicons/suicide_risk_lexicon_calibrated_unmatched_tokens_unvalidated_24-02-15T19-30-52.pickle")


feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(docs,
																						srl.constructs,normalize = normalize_lexicon, return_matches=True,
																						add_lemmatized_lexicon=True, lemmatize_docs=False,
																						exact_match_n = srl.exact_match_n,exact_match_tokens = srl.exact_match_tokens)


# =================================================================


i = 2
print(docs[i])
constructs_alphabetical = constructs_in_order.copy()
constructs_alphabetical.sort()
pd.DataFrame(matches_per_doc[i])[constructs_alphabetical]


# =================================================================




