# Author: Daniel Low
# License: see repository

# !pip install lightgbm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import re
import datetime
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.base import clone
import warnings
from sklearn.preprocessing import StandardScaler
# !pip install scikit-optimize
from skopt import BayesSearchCV # had to replace np.int for in in transformers.py
from construct_tracker import lexicon
from construct_tracker.machine_learning import metrics_report
from construct_tracker.machine_learning.feature_importance import generate_feature_importance_df, tfidf_feature_importances
from construct_tracker.machine_learning import pipelines




def create_binary_dataset(df_metadata, dv = 'suicide', n_per_dv = 3000, dv_location = 'columns'):
	"""
	
    Parameters
    ----------
    df_metadata : pd.DataFrame
        Dataframe containing columns for dv and text.
    dv : str
		Column or value name of dv
    n_per_dv : int
        Number of samples to take from each level of dv.
    dv_location : str
        Location of dv in df_metadata. Either 'subreddit' (or another column name containing DVs) or 'columns' (if dv is a column name).
    Returns
    -------
    df_metadata_tag : pd.DataFrame
        Binary dataset with equal number of samples in each level of dv.
    
	"""
	if dv_location == 'columns':

		df_metadata_tag_1 = df_metadata[df_metadata[dv]==1].sample(n=n_per_dv,random_state=123)
		df_metadata_tag_0 = df_metadata[df_metadata[dv]==0].sample(n=n_per_dv,random_state=123)
		assert df_metadata_tag_1.shape[0] == n_per_dv
		assert df_metadata_tag_0.shape[0] == n_per_dv

		df_metadata_tag = pd.concat([df_metadata_tag_1, df_metadata_tag_0]).sample(frac=1).reset_index(drop=True)

	else:	

		df_metadata_tag_1 = df_metadata[df_metadata[dv_location]==dv].sample(n=n_per_dv,random_state=123)
		df_metadata_tag_0 = df_metadata[df_metadata[dv_location]!=dv].sample(n=n_per_dv,random_state=123)
		assert df_metadata_tag_1.shape[0] == n_per_dv
		assert df_metadata_tag_0.shape[0] == n_per_dv

		df_metadata_tag = pd.concat([df_metadata_tag_1, df_metadata_tag_0]).sample(frac=1).reset_index(drop=True)
	return df_metadata_tag



# config
# ========================================================================================
output_dir = './data/output/reddit_binary_classification/'
output_dir_i = output_dir+'ml_performance/'
os.makedirs(output_dir , exist_ok=True)
os.makedirs(output_dir_i,exist_ok=True)

toy = False 
task = 'classification'


feature_vectors = [
				   'liwc22', 
				   'srl_validated',
				   ] # 'cts_token_clause', 'liwc22_semantic']#, ]#['all-MiniLM-L6-v2', 'srl_unvalidated','SRL GPT-4 Turbo', 'liwc22', 'liwc22_semantic'] # srl_unvalidated_text_descriptives','text_descriptives' ]
sample_sizes = {50, 150, 2000} # per group, so 100, 300, 4000 in

if task == 'classification':
	scoring = 'f1'
	metrics_to_report = 'all'
	model_names = ['LogisticRegression']
elif task == 'regression':
	scoring = 'neg_mean_squared_error'
	metrics_to_report = 'all'
	# metrics_to_report = ['Model','n', 'RMSE','RMSE per value','MAE','MAE per value',  'rho', 'gridsearch', 'Best parameters']
	model_names = ['LGBMRegressor', 'Ridge']
	
gridsearch = True # True, 'minority'
srl = lexicon.load_lexicon(name = 'srl_v1-0') # Load lexicon
srl_prototypical = lexicon.load_lexicon(name = 'srl_prototypes_v1-0') # Load lexicon
constructs_in_order = list(srl.constructs.keys())


# Subreddits for each construct
dv_tags = {
	'self_harm': 'selfharm',
 'suicide': 'SuicideWatch',
 'bully': 'bullying',
 'abuse_sexual': 'sexualassault',
 'bereavement': 'GriefSupport',
 'isolated': 'lonely',
 'anxiety': 'Anxiety',
 'depressed': 'depression',
 'gender': 'AskLGBT',
 'eating': 'EatingDisorders',
 'substance': 'addiction',
 }



srl_reddit_mapping =  {
 
 'Direct self-injury': 'selfharm',
 'Active suicidal ideation & suicidal planning': 'SuicideWatch',
 'Passive suicidal ideation': 'SuicideWatch',
 'Other suicidal language': 'SuicideWatch',
 'Bullying': 'bullying',
 'Sexual abuse & harassment': 'sexualassault',
 'Grief & bereavement': 'GriefSupport',
 'Loneliness & isolation': 'lonely',
 'Anxiety': 'Anxiety',
 'Depressed mood': 'depression',
 'Gender & sexual identity': 'AskLGBT',
 'Eating disorders': 'EatingDisorders',
 'Other substance use': 'addiction',
 'Alcohol use':'addiction',

 }


prompt_names = {'self_harm': 'self harm or self injury',
 'suicide': 'suicidal thoughts or suicidal behaviors',
 'bully': 'bullying',
 'abuse_sexual': 'sexual abuse',
 'bereavement': 'bereavement or grief',
 'isolated': 'loneliness or social isolation',
 'anxiety': 'anxiety',
 'depressed': 'depression',
 'gender': 'gender identity',
 'eating': 'an eating disorder or body image issues',
 'substance': 'substance use'}




# Load data
# ========================================================================================
import pickle



# load pickle
with open('./data/input/reddit/reddit_13_mental_health_4600_posts_20250311_123431_dfs.pkl', 'rb') as handle:
	dfs = pickle.load(handle)







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




# Model
# ========================================================================================
np.random.seed(123)
ts_i = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')

if toy:
	sample_sizes = [150]
	feature_vectors = feature_vectors[:2]
	dv_tags = {
		'self_harm': 'selfharm',
		'suicide': 'SuicideWatch',
	}




for i, n in enumerate(sample_sizes):
	
	results = []
	results_content_validity = []

	for feature_vector in feature_vectors:#['srl_unvalidated']:#, 'srl_unvalidated']:
		if 'cts' in feature_vector:
			n = 'zero_shot'
		if i >0:
			continue # skip because we don't have no training sets, it's zero shot
		
		if 'cts' not in feature_vector:
			train = dfs['train'][feature_vector].copy()
		test = dfs['test'][feature_vector].copy()
		X_test_df_content_validity = dfs['content_validity'][feature_vector].copy()
		X_test_df_content_validity

		if toy:
			output_dir_i = output_dir + f'results_{ts_i}_toy/'
		else:
			output_dir_i = output_dir + f'results_{ts_i}_{n}/'
			
		os.makedirs(output_dir_i, exist_ok=True)
		
		for dv in dv_tags.keys():
			print(dv)
			dv_reddit = dv_tags[dv]
			responses = []
			time_elapsed_all = []
			if 'cts' not in feature_vector:
				train_i = create_binary_dataset(train, dv = dv_reddit, n_per_dv = n, dv_location='subreddit')
				train_i[dv] = (train_i['subreddit']==dv_reddit)*1
				train_y = train_i[dv].tolist()
				train_i_y = train_i[['id', dv]]
				convo_ids_train = train_i['id'].values
				
			test_i = create_binary_dataset(test, dv = dv_reddit, n_per_dv = 300, dv_location='subreddit')
			test_i[dv] = (test_i['subreddit']==dv_reddit)*1
			test_y = test_i[dv].tolist()
			test_i_y = test_i[['id', dv]]
			convo_ids_test = test_i['id'].values

			try:
				train_i.rename(columns={'word_count_y': 'word_count'}, inplace=True)
				test_i.rename(columns={'word_count_y': 'word_count'}, inplace=True)
			except:
				pass
			if 'srl' in feature_vector:
				feature_names = constructs_in_order+['word_count']
			elif feature_vector == 'liwc22':
				feature_names = liwc_nonsemantic+liwc_semantic
			elif 'cts' in feature_vector:
				feature_names = [dv+'_max'] # TODO: try with all features 
			
			if 'cts' not in feature_vector:
				y_train =  train_i[dv].values
				X_train = train_i[feature_names]
			
				print(y_train.shape, X_train.shape)

			y_test =  test_i[dv].values
			X_test = test_i[feature_names]
			
			# content validity test sets
			
			X_test_df_content_validity['y_test_reddit']=X_test_df_content_validity['y_test'].map(srl_reddit_mapping)
			X_test_df_content_validity_dv = X_test_df_content_validity[X_test_df_content_validity['y_test_reddit']==dv_reddit][feature_names] # counts from lexicon
			y_test_3_dv = [1]*len(X_test_df_content_validity_dv)
						


			if toy:
				X_train['y'] = y_train
				X_train = X_train.sample(n = 100)
				y_train = X_train['y'].values
				X_train = X_train.drop('y', axis=1)

			
			
			for model_name in model_names: 
		
				pipeline = pipelines.get_pipelines(feature_vector, model_name = model_name)
				print(pipeline)
			
				if gridsearch == True:
					parameters = pipelines.get_params(feature_vector,model_name=model_name, toy=toy)
		
					pipeline = BayesSearchCV(pipeline, parameters, cv=5, scoring=scoring, return_train_score=False,
					n_iter=32, random_state=123)    
					if feature_vector != 'tfidf':
						if 'cts' not in feature_vector:
							if 'y' in X_train.columns:
								warnings.warn('y var is in X_train, removing')
								X_train = X_train.drop('y', axis=1)
							
							
					pipeline.fit(X_train,y_train)
					best_params = pipeline.best_params_
					best_model = pipeline.best_estimator_
					if feature_vector != 'tfidf':
						if 'y' in X_test.columns:
							warnings.warn('y var is in X_test, removing')
							X_test = X_test.drop('y', axis=1)
					# y_pred = best_model.predict(X_test)
					
					# Content validity
				

				else:
					pipeline.fit(X_train,y_train)
					best_params = 'No hyperparameter tuning'
					# y_pred = pipeline.predict(X_test)
				
				
			
				# Performance
				dv_clean = dv.replace(' ','_').capitalize()
				if task == 'classification':
					
					
					if gridsearch:
						y_proba = best_model.predict_proba(X_test)       # Get predicted probabilities
						# y_pred_content_validity_13 = best_model.predict(X_test_13_dv)
						y_pred_content_validity_3 = best_model.predict(X_test_df_content_validity_dv)
					else:

						y_proba = pipeline.predict_proba(X_test)       # Get predicted probabilities
						# y_pred_content_validity_13 = pipeline.predict(X_test_13_dv)
						y_pred_content_validity_3 = pipeline.predict(X_test_df_content_validity_dv)
					y_proba_1 = y_proba[:,1]
					y_pred = y_proba_1>=0.5*1                   # define your threshold
					# Predictions
					output_filename = f'{feature_vector}_{model_name}_{dv}_{n}'
					custom_cr, sklearn_cr, cm_df_meaning, cm_df, cm_df_norm, y_pred_df = metrics_report.save_classification_performance(y_test, y_pred, y_proba_1, output_dir_i, output_filename=output_filename,feature_vector=feature_vector, model_name=model_name,best_params = best_params, classes = ['Other', f'{dv_clean}'],amount_of_clauses=None, save_output=True)

					output_filename = f'content-validity-13_{feature_vector}_{model_name}_{dv}_{n}'
					# custom_cr_content_13, sklearn_cr, y_pred_df = metrics_report.save_classification_performance(y_test_13_dv, y_pred_content_validity_13, y_pred_content_validity_13, output_dir_i, output_filename=output_filename,feature_vector=feature_vector, model_name=model_name,best_params = best_params, classes = ['Other', f'{dv_clean}'],amount_of_clauses=None, save_confusion_matrix=False, save_output=True)
					output_filename = f'content-validity-3_{feature_vector}_{model_name}_{dv}_{n}'
					custom_cr_content_3, sklearn_cr, y_pred_df = metrics_report.save_classification_performance(y_test_3_dv, y_pred_content_validity_3, y_pred_content_validity_3, output_dir_i, output_filename=output_filename,feature_vector=feature_vector, model_name=model_name,best_params = best_params, classes = ['Other', f'{dv_clean}'],amount_of_clauses=None, save_confusion_matrix=False, save_output=True)
																														 

				elif task == 'regression':
					if gridsearch:
						y_pred = best_model.predict(X_test)
					else:
						y_pred = pipeline.predict(X_test)

					results_i = metrics_report.regression_report(y_test,y_pred,y_train=y_train,
											metrics_to_report = metrics_to_report,
												gridsearch=gridsearch,
											best_params=best_params,feature_vector=feature_vector,model_name=model_name, plot = True, save_fig_path = path,n = n, round_to = 2)
				
				results.append(custom_cr)
				
				results_content_validity.append(custom_cr_content_3)
				
				# Feature importance
				if feature_vector == 'tfidf':
					if model_name in ['XGBRegressor']:
						warnings.warn('Need to add code to parse XGBoost feature importance dict')
					else:
						feature_importances = tfidf_feature_importances(pipeline, top_k = 50, savefig_path = output_dir_i + f'feature_importance_{feature_vector}_{model_name}_{n}_{ts_i}_{dv}')
				else:
					feature_names = X_train.columns
					
					feature_importance = generate_feature_importance_df(pipeline, model_name,feature_names,  xgboost_method='weight', model_name_in_pipeline = 'model')
					if str(feature_importance) != 'None':       # I only implemented a few methods for a few models
						feature_importance.to_csv(output_dir_i + f'feature_importance_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}_{dv}.csv', index = False)        
				
					
				
	results_df = pd.concat(results)
	results_df = results_df.reset_index(drop=True)
	results_df.to_csv(output_dir_i + f'results_{n}_{ts_i}.csv', index=False)

	results_df_content_validity = pd.concat(results_content_validity)
	results_df_content_validity = results_df_content_validity.reset_index(drop=True)
	results_df_content_validity.to_csv(output_dir_i + f'results_content_validity_{n}_{ts_i}.csv', index=False)


	
	# NaN analysis
	if type(X_train) == pd.core.frame.DataFrame:
		df = X_train.copy()
		# Find the column and index of NaN values
		nan_indices = df.index[df.isnull().any(axis=1)].tolist()
		nan_columns = df.columns[df.isnull().any()].tolist()
		# print("Indices of NaN values:", nan_indices)
		print("Columns with NaN values:", nan_columns)
		print(df.size)
		nans = df.isna().sum().sum()
		print('% of nans:', np.round(nans/df.size,3))
	



# cts
# ============================================================

feature_vectors = ['cts_single',
				   'cts_multi',]

ts_i = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
results = []
results_content_validity = []

best_params = None

dv_tags_r = {v: k for k, v in dv_tags.items()}


n = 0
for feature_vector in feature_vectors:#['srl_unvalidated']:#, 'srl_unvalidated']:
	test = dfs['test'][feature_vector].copy()
	test['subreddit_mapping'] = test['subreddit'].map(dv_tags_r)
	X_test_3 = dfs['content_validity'][feature_vector].copy()
	
	if toy:
		output_dir_i = output_dir + f'results_{ts_i}_toy/'
	else:
		output_dir_i = output_dir + f'results_{ts_i}_{feature_vector}/'
		
	os.makedirs(output_dir_i, exist_ok=True)
	



	for dv in dv_tags:
		
		responses = []
		time_elapsed_all = []

		test_i = create_binary_dataset(test, dv = dv, n_per_dv = 300, dv_location = 'subreddit_mapping')
		test_i['subreddit'].value_counts()
		y_test =  test_i['subreddit_mapping'].values
		y_test = [1 if n == dv else 0 for n in y_test]

		test_i_y = test_i[['id', dv+'_max']]
		# convo_ids_test = test_i['id'].values

		feature_names = [dv+'_max'] # TODO: try with all features 

		y_proba_1 = test_i[feature_names].values
		y_proba_1 = [n[0] for n in y_proba_1]
		
		optimal_threshold = np.round(metrics_report.find_optimal_threshold(y_test, y_proba_1),3)
		threshold_05 = 0.5

		# content validity
		y_proba_1_3_dv = X_test_3[X_test_3['y_test_reddit']==dv_tags.get(dv)][feature_names].values
		y_proba_1_3_dv = [n[0] for n in y_proba_1_3_dv]
		assert y_proba_1_3_dv != []
		y_test_3_dv = [1]*len(y_proba_1_3_dv)
		y_pred_3_dv = y_proba_1_3_dv>=optimal_threshold
		y_pred_3_dv = [n*1 for n in y_pred_3_dv]
		
		
		for threshold in [threshold_05, optimal_threshold]:
			y_pred = y_proba_1>=optimal_threshold*1                   # define your threshold
			y_pred = [n*1 for n in y_pred]			
			dv_clean = dv.replace('_',' ').capitalize()
			threshold_clean = str(threshold).replace('.', '')
			output_filename = f'{feature_vector}_{dv_clean}_{n}_clauses-all_threshold-{threshold_clean}'
			custom_cr, sklearn_cr, cm_df_meaning, cm_df, cm_df_norm, y_pred_df = metrics_report.save_classification_performance(y_test, y_pred, y_proba_1, output_dir_i, 
																							output_filename=output_filename,feature_vector=feature_vector, model_name=None,best_params = None, classes = ['Other', f'{dv_clean}'],amount_of_clauses=None, save_output=True)
			
			output_filename = f'content-validity-3_{feature_vector}_{dv_clean}_{n}_clauses-all_threshold-{threshold_clean}'
			custom_cr_content_validity, _, y_pred_df = metrics_report.save_classification_performance(y_test_3_dv, y_pred_3_dv, y_proba_1_3_dv, output_dir_i, 
																							output_filename=output_filename,feature_vector=feature_vector, model_name=None,best_params = None, classes = ['Other', f'{dv_clean}'],amount_of_clauses=None, save_confusion_matrix = False, save_output=True)	

		results.append(custom_cr)
		# results_content_validity.append(results_i_content_13)
		results_content_validity.append(custom_cr_content_validity)

				
	results_df = pd.concat(results)
	results_df = results_df.reset_index(drop=True)
	results_df.to_csv(output_dir_i + f'results_{n}_{ts_i}.csv', index=False)

	results_df_content_validity = pd.concat(results_content_validity)
	results_df_content_validity = results_df_content_validity.reset_index(drop=True)
	results_df_content_validity.to_csv(output_dir_i + f'results_content_validity_{n}_{ts_i}.csv', index=False)








