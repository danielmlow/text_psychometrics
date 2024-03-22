#!/usr/bin/env python
# coding: utf-8


# Note: see construct_text_similarity.py for more details on preprocessing

import pandas as pd
import numpy as np
import os
import datetime
import re
import seaborn as sns
import sys
# local scripts
sys.path.append( './../concept-tracker/')
from concept_tracker.utils import stop_words
from concept_tracker.utils import clean
from concept_tracker.utils.tokenizer import spacy_tokenizer
from concept_tracker.utils.lemmatizer import spacy_lemmatizer
from concept_tracker import cts

ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')


# ============================================================


input_dir = './../../../data/ctl/input/datasets/'
output_dir = './data/'

task = 'classification'


# ============================================================


import pickle
run_this = False#True saves, False loads
if run_this:
    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'wb') as f:
        pickle.dump(dfs, f) 
else:

    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'rb') as f:
    	dfs = pickle.load(f)


# ============================================================


# config
construct_representation = 'lexicon' #(lexicon is good for 'prototypes')
doc_representation = 'clauses'
embedding_type = 'sentence'
embedding_model = 'all-MiniLM-L6-v2'
# embedding_package = 'flair'


# # Load data
# ### TODO: remove words with negation 1-3 words prior
# ============================================================

entire_test_set = True
balanced = False

train = pd.read_parquet(input_dir + f'train10_train_metadata_messages_clean.gzip', engine='pyarrow')
X_train = pd.read_csv('./data/input/ctl/X_train_all_with_interaction_preprocessed_24-03-07T04-25-04.csv', index_col = 0) # preprocessing


test = pd.read_parquet(input_dir + f'train10_test_metadata_messages_clean.gzip', engine='pyarrow')
X_test = pd.read_csv('./data/input/ctl/X_test_all_with_interaction_preprocessed_24-03-07T04-25-04.csv', index_col = 0) # preprocessing

if entire_test_set:
	# All test samples
	test['suicide_ladder_8'].value_counts() 
	# see get_true_risk_8 function in train_test_split.ipynb for interpretation
	active_rescue = test[test['suicide_ladder_8']==8] # immiment risk which could not be de-escaleted
	imminent_risk = test[test['suicide_ladder_8'].isin([6,7])] # has intent and capability and timeframe
	timeframe = imminent_risk[imminent_risk['timeframe']==1] # has intent and capability and timeframe
	suicidal = test[test['suicide_ladder_8'].isin([2])] # no imminent risk, no intent or capability; desire or other forms of suicide without any of those tags
	suicidal_desire = suicidal[suicidal['suicidal_desire']==1] # no imminent risk, no intent or capability; desire confirmed
	if balanced:
		# subselect to match minority class:
		# ============================================================
		# TODO: you cna take 50 samples away from each group and keep test set imbalanced
		# To start, do balanced and train a model 50 each. 
		# active_rescue
		smallest_group = active_rescue.shape[0]
		timeframe = timeframe.sample(n= smallest_group, random_state=123)
		suicidal_desire = suicidal_desire.sample(n= smallest_group, random_state=123)
		df = pd.concat([active_rescue, timeframe, suicidal_desire]).sample(frac=1).reset_index(drop=True)
	
else:
	# All X_test samples
	X_test = X_test.merge(test, on='conversation_id', how='left')
	X_test['suicide_ladder_8'].value_counts()
	imminent_risk = X_test[X_test['suicide_ladder_8'].isin([6,7])] # has intent and capability and timeframe
	timeframe = imminent_risk[imminent_risk['timeframe']==1] # has intent and capability and timeframe
	suicidal = X_test[X_test['suicide_ladder_8'].isin([2])] # no imminent risk, no intent or capability; desire or other forms of suicide without any of those tags
	suicidal_desire = suicidal[suicidal['suicidal_desire']==1] # no imminent risk, no intent or capability; desire confirmed
	if balanced:
		# subselect to match minority class:
		# ============================================================
			# just work with these two
		smallest_group = timeframe.shape[0]
		suicidal_desire = suicidal_desire.sample(n= smallest_group, random_state=123)
	df = pd.concat([timeframe, suicidal_desire]).sample(frac=1).reset_index(drop=True)





	













# TODO confirm no capability or intent in desire group





"""
def get_true_risk_8(row):
	if (row['3rd_party'] ==1 or row['testing'] == 1 or row['prank'] == 1):
		return -1
	elif (row['active_rescue'] > 0):
		return 8 # active rescue
	
	elif (row['ir_flag'] > 0):
		return 7 # high risk
	
	elif (row['timeframe'] > 0):
		return 6 # high risk
	
	elif (row['suicidal_capability'] > 0):
		return 5 # high risk
	
	elif (row['suicidal_intent']>0):
		return 4
	elif row['self_harm']>0:
		return 3

	elif (row['suicidal_desire']>0 or row['suicide']>0):
		return 2
	else: 
		return 1

		"""



# # Encode embeddings and compute similarity
# ============================================================
# ### Construct (Lexicon prototypes)
# ============================================================

import dill
import srl_constructs
srl = dill.load(open("./../lexicon/data/input/lexicons/suicide_risk_lexicon_validated_prototypical_tokens_24-03-06T00-47-30.pickle", "rb"))
constructs_to_measure = srl_constructs.constructs_in_order

# # Encode lexicon
# you want to encode each token once, because can appear in multiple lexicons
# ============================================================

embeddings_dir = './../lexicon/data/input/lexicons/'
prior_embeddings = dill.load(open(embeddings_dir+'embeddings_lexicon-tokens_all-MiniLM-L6-v2.pickle', "rb"))
# import tensorboard
from sentence_transformers import SentenceTransformer
embeddings_name = 'all-MiniLM-L6-v2'
sentence_embedding_model = SentenceTransformer(embeddings_name)       # load embedding

# dictionary split by construct
construct_tokens_d = {}
for construct in srl.constructs.keys():
	tokens = srl.constructs[construct]['tokens']                      
	construct_tokens_d[construct] = tokens

# single dictionary for all tokens, not split by construct
construct_embeddings_d = {}
tokens_to_encode = []
for construct in srl.constructs.keys():
	tokens = srl.constructs[construct]['tokens']                      
	for token_i in tokens:
		if token_i in prior_embeddings.keys():
			embedding = prior_embeddings[token_i]
			construct_embeddings_d[token_i] = embedding
		else:
			tokens_to_encode.append(token_i)



# ============================================================
# Encoding docs
# ============================================================
# 100m 6000 with interaction
with open('./data/input/ctl/embeddings/'+f'embeddings_all-MiniLM-L6-v2_docs_clauses_with-interaction_24-03-07T04-25-04.pickle', 'rb') as handle:
		docs_embeddings_d = pickle.load(handle)




# # 
# # ============================================================
docs_to_encode = []
docs_embeddings_d_subset = {}
for conversation_id in df['conversation_id'].values:
	if conversation_id in docs_embeddings_d.keys():
		embedding = docs_embeddings_d[conversation_id]
		docs_embeddings_d_subset[conversation_id] = embedding
	else:
		docs_to_encode.append(conversation_id)
len(docs_to_encode)	


if len(docs_to_encode)>0:
	# TODO need to clean this up, won't work as is

	# tokenized into clauses 
	# ============================================================
	# 15m samples for 38k CTL convos just texter
	# 11m samples for 6500 CTL convos  texter+counselor

	run_this = True

	if run_this:

		print(len(docs_clean))

		docs_clean = [n.replace('\n', '. ') for n in docs_clean] # help tokenize by clause
		# docs_clean_clauses = [clean.remove_multiple_spaces(doc) for doc in docs_clean]
		docs_clean_clauses = spacy_tokenizer(docs_clean, 
										language = 'en', model='en_core_web_sm', 
										method = 'clause', # clause tokenization
										lowercase=False, 
										display_tree = False, 
										remove_punct=False, 
										clause_remove_conj = True)
		
		
		df['docs_clean_clauses'] = docs_clean_clauses

		df.to_csv(f'./data/input/ctl/X_test_all_with_interaction_preprocessed_{ts}.csv')


	def clean_ctl_conversation(docs):

		docs_clean_clauses_clean = []
		for doc in docs_clean_clauses:
			clauses_doc_i = [] 
			for clause in doc:
				clauses_doc_i.extend(clause.split('\n'))
			doc_clean = [n.replace('texter : ','').replace('counselor : ','').replace('\n','. ').strip('.,:\n').replace(" '", "'").replace(' ’', "'").replace(' ,', ',').strip(' ').replace('observer : ', '').replace(" n't", "n't").replace(" ( 1/2 )", "").replace('{ { URL } }', '').replace('[ scrubbed ]','').replace('  ', ' ') for n in clauses_doc_i]
			doc_clean = [n for n in doc_clean if (n not in ['texter', 'counselor', 'observer', '', '-- UNREADABLE MESSAGE --', '( 2/2 )', '( 1/2 )']) and (len(n)>5)]
			
			docs_clean_clauses_clean.append(doc_clean)
		return docs_clean_clauses_clean





	docs_clean_clauses_clean = clean_ctl_conversation(docs_clean_clauses)


	# import tensorboard
	from sentence_transformers import SentenceTransformer
	embeddings_name = 'all-MiniLM-L6-v2'
	sentence_embedding_model = SentenceTransformer(embeddings_name)       # load embedding
	# sentence_embedding_model._first_module().max_seq_length = 500
	print(sentence_embedding_model .max_seq_length) #default = 256

	test['docs_clean_clauses']

	for doc in docs_to_encode:
		doc_clauses = df[df['conversation_id'] == doc]['docs_clean_clauses'].values
	embeddings = sentence_embedding_model.encode(tokens_to_encode, convert_to_tensor=True,show_progress_bar=True)	
	embeddings_d = dict(zip(tokens_to_encode, embeddings))
	prior_embeddings.update(embeddings_d)



	len(docs_embeddings_d.keys())
	len(docs_embeddings_d_subset.keys())




# CTS similarity
len(docs_embeddings_d_subset.keys())

docs_embeddings_d_subset2 = {}
for k, v in docs_embeddings_d_subset.items():
	if k not in feature_vectors['doc_id'].astype(int).values:
		docs_embeddings_d_subset2[k] = v


len(docs_embeddings_d_subset2.keys())

import time

# TODO: dont use minmax is splitting afterwards

start = time.time()
method = 'lexicon_clause'
feature_vectors2, cosine_scores_docs2 = cts.measure(
			construct_tokens_d = construct_tokens_d,
			construct_embeddings_d = construct_embeddings_d,
			docs_embeddings_d = docs_embeddings_d_subset2,
			method = method, 
			summary_stat = ['max'],
			return_cosine_similarity=True,
			minmaxscaler = (0,1)
		)



feature_vectors3 = pd.concat([feature_vectors, feature_vectors2])
feature_vectors3.drop_duplicates(subset = ['doc_id'], inplace=True)


end = time.time()
print(end - start)

# TODO: Plot histogram for both groups before and after
feature_vectors.describe()

df.rename(columns={'doc_id':'conversation_id'}, inplace=True)
df = df.merge(feature_vectors3,on='conversation_id')
df.columns = [n.replace('_max', '') for n in df.columns]

# train-test split 50-150
from sklearn.model_selection import train_test_split
import srl_constructs
X_df = df[srl_constructs.constructs_in_order]
y_df = df['timeframe'].values
df['timeframe'].value_counts()
y_df.shape[0]*0.15
X_train, X_test, y_train, y_test = train_test_split(X_df,y_df, train_size=0.6, random_state=12)

X_train.shape
from collections import Counter
Counter(y_train)
Counter(y_test)
X_test.shape

gridsearch = False

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier # TODO: add

model = LogisticRegression(random_state=0)
model_name = 'LogisticRegression'

model = LGBMClassifier(random_state=0)
model_name = 'LGBMClassifier'

feature_vector = 'cts_49_srl_prototypes'
pipeline = Pipeline([
			('imputer', SimpleImputer(strategy='median')),
			('standardizer', StandardScaler()),
			 ('model', model), 
			])

if gridsearch == True:
	parameters = get_params(feature_vector,model_name=model_name, toy=toy)

	pipeline = BayesSearchCV(pipeline, parameters, cv=5, scoring=scoring, return_train_score=False,
	n_iter=32, random_state=123)    
	if feature_vector != 'tfidf':
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


output_dir_i = output_dir+'output/desire_vs_imminent_risk/'
os.makedirs(output_dir_i,exist_ok=True)

import sys
sys.path.append( './../../concept-tracker/')
from concept_tracker.utils.metrics_report import cm, custom_classification_report, regression_report, generate_feature_importance_df


if gridsearch:
	y_proba = best_model.predict_proba(X_test)       # Get predicted probabilities
	y_pred_content_validity_13 = best_model.predict(X_test_13_dv)
	y_pred_content_validity_3 = best_model.predict(X_test_3_dv)
else:

	y_proba = pipeline.predict_proba(X_test)       # Get predicted probabilities
	
y_proba_1 = y_proba[:,1]
y_pred = y_proba_1>=0.5*1                   # define your threshold
# Predictions
y_pred_df = pd.DataFrame(y_pred)

n = 'full'
y_pred_df.to_csv(output_dir_i+f'y_pred_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}_{dv}.csv', index=False)
path = output_dir_i + f'{feature_vector}_{model_name}_{n}_{ts_i}_{dv}'

dv_clean = dv.replace('_',' ')
cm_df_meaning, cm_df, cm_df_norm = cm(y_test,y_pred, output_dir_i, f'{model_name}_{dv}', ts_i, classes = ['Other', f'{dv_clean}'], 
	save=False)

dv = 'timeframe'
ts_i = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
results_i = custom_classification_report(y_test, y_pred, y_proba_1, output_dir_i,gridsearch=gridsearch,
						best_params=best_params,feature_vector=feature_vector,model_name=f'{model_name}_{dv}',round_to = 2, ts = ts_i)




feature_names = X_train.columns
# TODO add correlation with DV to know direction
feature_importance = generate_feature_importance_df(pipeline, model_name,feature_names,  xgboost_method='weight', model_name_in_pipeline = 'model')
if str(feature_importance) != 'None':       # I only implemented a few methods for a few models
feature_importance.to_csv(output_dir_i + f'feature_importance_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}_{dv}.csv', index = False)        
# display(feature_importance.iloc[:50])












for dv_0_1 in [
	['suicidal_desire', 'immin'],
	# ['timeframe', 'active_rescue']
	]:
	pass
	
	# create df only with these two
	df.shape
	df_i = df[(df[dv_0_1[0]] == 1) | (df[dv_0_1[1]] == 1)]
	feature_vectors_i = feature_vectors[feature_vectors['doc_id'].isin(df_i['conversation_id'])]
	

	# Repeat process of extracting but just looking at first N clauses. 











# content validity test sets: extract features on these tokens
# ===================================
for split in ['X_test_content_validity_prototypicality-1_3', 'X_test_content_validity_prototypicality-3']:
# for split in ['X_test_content_validity_prototypicality-3']:
	df_i = dfs[split]['df_text'].copy()
	df_i = df_i.reset_index()
	df_i_index = df_i.index.values
	print(df_i.shape)
	docs = df_i['token'].tolist()

	tokens_to_encode = [n for n in docs if n not in prior_embeddings.keys()]
	if len(tokens_to_encode)>0:
		embeddings = sentence_embedding_model.encode(tokens_to_encode, convert_to_tensor=True,show_progress_bar=True)	
		embeddings_d = dict(zip(tokens_to_encode, embeddings))
		prior_embeddings.update(embeddings_d)
	
	docs_embeddings = [[prior_embeddings[token]] for token in docs]
	assert len([n for n in docs_embeddings if n is  None]) == 0 # if not encode

	embeddings_tokens_docs_content_validity_d = dict(zip(df_i_index,docs_embeddings ))
	method = 'lexicon_clause'
	feature_vectors_i_wc_content, cosine_scores_docs_i_wc_content = cts.measure(
			construct_tokens_d = construct_singletoken_tokens_d,
			construct_embeddings_d = construct_singletoken_embeddings_d,
			# docs = docs_clean_clauses,
			docs_embeddings_d = embeddings_tokens_docs_content_validity_d,
			method = method, #todo: change to token, tokens, weighted_tokens
			summary_stat = ['max'],
			return_cosine_similarity=True,
			minmaxscaler = (0,1)
		)
	
	if df_i.shape[0]==feature_vectors_i_wc_content.shape[0]:
		df_i[feature_vectors_i_wc_content.columns] = feature_vectors_i_wc_content.values
	
	dfs[split]['cts_token_clause'] = df_i.copy()

# Save 
run_this = False #True saves, False loads
if run_this:
	# Save
    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'wb') as f:
        pickle.dump(dfs, f) 
else:
	# Load
    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'rb') as f:
    	dfs = pickle.load(f)










# Evaluate on test set
# ============================================================
from sklearn.metrics import roc_curve
import numpy as np

def find_optimal_threshold(y_true, y_pred):
    """
    Find the optimal threshold for binary classification based on Youden's Index.
    
    Parameters:
    y_true (array-like): True binary labels.
    y_pred (array-like): Target scores, probabilities of the positive class.
    
    Returns:
    float: Optimal threshold value.
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # Compute Youden's J index
    youdens_j = tpr - fpr
    
    # Find the optimal threshold
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold

# Example usage





output_dir_i = output_dir+'ml_performance/'
os.makedirs(output_dir_i,exist_ok=True)


# 64,51,54 vs .4, .25, 56 (with much more training data)


np.random.seed(123)

# TODO: see where to save feature_vector (tfidf, liwc22) and where to save model_name (Ridge, LightGBM)
import dill
def load_lexicon(path):
	lexicon = dill.load(open(path, "rb"))
	return lexicon
srl = load_lexicon("./../lexicon/data/input/lexicons/suicide_risk_lexicon_validated_24-03-06T00-37-15.pickle")
constructs_in_order = list(srl.constructs.keys())



ts_i = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')


def create_binary_dataset(df_metadata, dv = 'suicide', n_per_dv = 3000):
	df_metadata_tag_1 = df_metadata[df_metadata[dv]==1].sample(n=n_per_dv,random_state=123)
	df_metadata_tag_0 = df_metadata[df_metadata[dv]==0].sample(n=n_per_dv,random_state=123)
	assert df_metadata_tag_1.shape[0] == n_per_dv
	assert df_metadata_tag_0.shape[0] == n_per_dv

	df_metadata_tag = pd.concat([df_metadata_tag_1, df_metadata_tag_0]).sample(frac=1).reset_index(drop=True)

	return df_metadata_tag


location = 'local'


if location == 'openmind':
  input_dir = '/nese/mit/group/sig/projects/dlow/ctl/datasets/'
  output_dir = 'home/dlow/zero_shot/data/output/'
elif location =='local':
  input_dir = '/Users/danielmlow/data/ctl/input/datasets/'
  output_dir = './data/output/'
os.makedirs(output_dir, exist_ok=True)


test = pd.read_parquet(input_dir + f'train10_test_metadata_messages_clean.gzip', engine='pyarrow')
test.shape


toy = False



feature_vectors = [
	# 'cts_token_clause',
				   'cts_prototypes_clause' 
				   #'liwc22', 
				#    'srl_validated'
				   
				   ] #, 'liwc22_semantic']#, ]#['all-MiniLM-L6-v2', 'srl_unvalidated','SRL GPT-4 Turbo', 'liwc22', 'liwc22_semantic'] # srl_unvalidated_text_descriptives','text_descriptives' ]
sample_sizes = [50] 

task = 'classification'
if task == 'classification':
	scoring = 'f1'
	metrics_to_report = 'all'
	model_names = ['LogisticRegression']
	
elif task == 'regression':
	scoring = 'neg_mean_squared_error'
	# metrics_to_report = ['Model','n', 'RMSE','RMSE per value','MAE','MAE per value',  'rho', 'gridsearch', 'Best parameters']
	model_names = ['LGBMRegressor', 'Ridge']
	metrics_to_report = 'all'

gridsearch = True#, 'minority'
# balance = True
output_dir = './data/output/binary_classification/'
os.makedirs(output_dir , exist_ok=True)





toy = False
if toy:
	sample_sizes = [50]
	feature_vectors = feature_vectors[:2]


results = []
results_content_validity = []
# for gridsearch in [True]:
from concept_tracker.utils import metrics_report

best_params = None

n = 0

# for feature_vector in ['srl_unvalidated', 'all-MiniLM-L6-v2']:#['srl_unvalidated']:#, 'srl_unvalidated']:
for feature_vector in feature_vectors:#['srl_unvalidated']:#, 'srl_unvalidated']:
	X_test_3 = dfs['X_test_content_validity_prototypicality-3'][feature_vector]
	# X_test_13 = dfs['X_test_content_validity_prototypicality-1_3'][feature_vector]
	

	if toy:
		output_dir_i = output_dir + f'results_{ts_i}_toy/'
	else:
		output_dir_i = output_dir + f'results_{ts_i}_{feature_vector}/'
		
	os.makedirs(output_dir_i, exist_ok=True)
	
	for dv in ctl_tags13:
		
		
		
		
		responses = []
		
		time_elapsed_all = []
	
	

		test_i = create_binary_dataset(test, dv = dv, n_per_dv = 300)
		y_test =  test_i[dv].values
		test_i_y = test_i[['conversation_id', dv]]
		convo_ids_test = test_i['conversation_id'].values

		
		feature_names = [dv+'_max'] # TODO: try with all features 
		
	
		
		# X_train_df_features = dfs['train'][feature_vector].copy()
		# X_train_df_features =  X_train_df_features[X_train_df_features['conversation_id'].isin(convo_ids_train)]
		# X_train_df_features = X_train_df_features.drop(dv,axis=1)
		# train_i_y_X = pd.merge(train_i_y, X_train_df_features, on='conversation_id')
		# y_train =  train_i_y_X[dv].values
		# X_train = train_i_y_X[feature_names]
		
		# print(y_train.shape, X_train.shape)


		X_test_df_features = dfs['test'][feature_vector].copy()
		# rename doc_id
		X_test_df_features = X_test_df_features.rename(columns={'doc_id':'conversation_id'})
		X_test_df_features =  X_test_df_features[X_test_df_features['conversation_id'].isin(convo_ids_test)]

		# X_test_df_features = X_test_df_features.drop(dv,axis=1)
		test_i_y_X = pd.merge(test_i_y,X_test_df_features, on='conversation_id')
		y_test =  test_i_y_X[dv].values
		y_proba_1 = test_i_y_X[feature_names].values

		
		
		# content validity test sets
		# y_proba_1_13_dv = X_test_13[X_test_13['y_test']==dv][feature_names].values
		# y_proba_1_13_dv = [n[0] for n in y_proba_1_13_dv]
		# y_test_13_dv = [1]*len(y_proba_1_13_dv)


		y_proba_1_3_dv = X_test_3[X_test_3['y_test']==dv][feature_names].values
		y_proba_1_3_dv = [n[0] for n in y_proba_1_3_dv]
		y_test_3_dv = [1]*len(y_proba_1_3_dv)
		from sklearn import metrics
		roc_auc = metrics.roc_auc_score(y_test, y_proba_1)
		

		optimal_threshold = np.round(find_optimal_threshold(y_test, y_proba_1),3)
		threshold_05 = 0.5
		
		for threshold in [threshold_05, optimal_threshold]:
			
			y_pred = y_proba_1>=optimal_threshold*1                   # define your threshold
			y_pred = [n[0]*1 for n in y_pred]
			# from concept_tracker.utils.metrics_report import cm, custom_classification_report
			# from importlib import reload
			# from concept_tracker.utils import metrics_report
			
			dv_clean = dv.replace('_',' ').capitalize()
			threshold_clean = str(threshold).replace('.', '')
			cm_df_meaning, cm_df, cm_df_norm = metrics_report.cm(y_test,y_pred, output_dir_i, f'{feature_vector}_{dv}_thesh-{threshold_clean}', ts_i, classes = ['Other', f'{dv_clean}'], save=True)
			# TODO: change to compute ROC AUC with y_proba
			results_i = metrics_report.custom_classification_report(y_test, y_pred, y_proba_1, output_dir_i,gridsearch=gridsearch,
									best_params=None,feature_vector=feature_vector,model_name=f'{feature_vector}_{dv}_thesh-{threshold_clean}',round_to = 2, ts = ts_i)
		
		reload(metrics_report)
		# Predictions
		y_pred_df = pd.DataFrame(y_proba_1)
		model_name = feature_vector
		
		y_pred_df.to_csv(output_dir_i+f'y_pred_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}_{dv}.csv', index=False)
		# pd.DataFrame(y_proba_1_13_dv).to_csv(output_dir_i+f'y_proba_13_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}_{dv}_thesh-{threshold_clean}.csv', index=False)
		pd.DataFrame(y_proba_1_3_dv).to_csv(output_dir_i+f'y_proba_3_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}_{dv}_thesh-{threshold_clean}.csv', index=False)

		
		# path = output_dir_i + f'{feature_vector}_{model_name}_{n}_{ts_i}_{dv}'
		# cm_df_meaning, cm_df, cm_df_norm = metrics_report.cm(list(y_test),y_pred, output_dir_i, f'{model_name}_{dv}_thesh-{threshold_clean}', ts_i, classes = ['Other', f'{dv_clean}'], save=True)
		# results_i = custom_classification_report(y_test, y_pred, y_proba_1, output_dir_i,gridsearch=gridsearch,
		# 						best_params=best_params,feature_vector=feature_vector,model_name=f'{model_name}_{dv}_thesh-{threshold_clean}',round_to = 2, ts = ts_i)
		
		
		# y_pred_13_dv = y_proba_1_13_dv>=optimal_threshold
		# y_pred_13_dv = [n*1 for n in y_pred_13_dv]
		y_pred_3_dv = y_proba_1_3_dv>=optimal_threshold
		y_pred_3_dv = [n*1 for n in y_pred_3_dv]
		# metrics.recall_score(y_test_13_dv, y_proba_1_13_dv)
		# results_i_content_13 = metrics_report.custom_classification_report(y_test_13_dv, y_pred_13_dv, y_pred_13_dv, output_dir_i,gridsearch=gridsearch,
		# 						best_params=best_params,feature_vector=feature_vector,model_name=f'{model_name}_{dv}_content-validity-13_thesh-{threshold_clean}',round_to = 2, ts = ts_i)
		results_i_content_3 = metrics_report.custom_classification_report(y_test_3_dv, y_pred_3_dv, y_pred_3_dv, output_dir_i,gridsearch=gridsearch,
								best_params=best_params,feature_vector=feature_vector,model_name=f'{feature_vector}_{model_name}_{dv}_content-validity-3_thesh-{threshold_clean}',round_to = 2, ts = ts_i)


		results.append(results_i)
		# results_content_validity.append(results_i_content_13)
		results_content_validity.append(results_i_content_3)

				
	results_df = pd.concat(results)
	results_df = results_df.reset_index(drop=True)
	results_df.to_csv(output_dir_i + f'results_{n}_{ts_i}.csv', index=False)

	results_df_content_validity = pd.concat(results_content_validity)
	results_df_content_validity = results_df_content_validity.reset_index(drop=True)
	results_df_content_validity.to_csv(output_dir_i + f'results_content_validity_{n}_{ts_i}.csv', index=False)









