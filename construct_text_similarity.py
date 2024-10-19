#!/usr/bin/env python
# coding: utf-8

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
# from concept_tracker.utils.embeddings import vectorize
from concept_tracker import cts

ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
ts



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

# ============================================================




X_test_df = pd.read_csv('./data/input/ctl/X_test_all_with_interaction.csv')
X_test_df = X_test_df.iloc[:,:2]

df = X_test_df.copy()
text_col = 'message_with_interaction_clean'


# ============================================================


# approach_embedding_names = [  # lexicon centroid weighted by distance to construct label
#    # 'w_w_glove', 
#     # 'w_c_psychbert',
#     # 'w_w_minilm', 
#     'w_c_minilm',           
#     # 'wl_w_minilm',
#     # 'wl_c_minilm'
# ]
	

# embedding_name_type = {
# #    model_name:embedding_type 
#     # 'glove': 'word',
#     'all-MiniLM-L6-v2': 'sentence',
#     # 'all-MiniLM-L6-v2': 'document',
#     # 'mnaylor/psychbert-cased': 'document',# need to fix  
# }


	


# ============================================================


# len(docs)


# # Preprocessing
# 
# 1. spell checker: automatically may create a lot of errors. It works by prividing closest orthographic neighbor to words not in a dictionary pyspellchecker is an option.
# 2. Remove authentification words: TALK, CONNECT, FEEL, BREATHE, HOPELINE, (can be lower case?)
# 

# ### TODO: remove words with negation 1-3 words prior

# ============================================================




# Fast: 1 sec every 10 000 messages

run_this = True

if run_this:
	docs =df[text_col].values
	# docs = [re.sub("[\(\[].*?[\)\]]", "", n) for n in docs] #replace text within parentheses/brackets and parentheses/brackets
	# docs = [n.replace('//', '').replace(' .', '.').replace(' ,', ',') for n in docs] 
	# docs = [n.replace('ampx200b', '').replace('\n','').replace('\xa0', '') for n in docs]
	docs_clean = [str(n) if str(n)!='nan' else '' for n in docs]
	docs_clean = [n.replace('!.', '!').replace('?.', '?').replace('....', '...').replace('...', '... ') for n in docs_clean]
	docs_clean = [clean.remove_multiple_spaces(doc) for doc in docs_clean]
	df['docs_clean'] = docs_clean
	print('made a difference in', df[df[text_col] != df['docs_clean']].shape[0]/df.shape[0], 'samples')




# tokenize documents into words (remove stop words and lemmatize)
# ============================================================

run_this=False
if run_this:
	more_stop_words = ['ca', 'nt','like', "'", "´", "n’t"]
	# words: tokenize by words, remove stop words and lemmatize for word-word similarity
	docs_clean_w_w = stop_words.remove(list(docs_clean), language = 'en', sws = 'nltk', remove_punct=True, extend_stopwords=more_stop_words)
	docs_clean_w_w = [clean.remove_multiple_spaces(doc) for doc in docs_clean_w_w]
	docs_clean_w_w = spacy_lemmatizer(docs_clean_w_w, language ='en') #this takes 22s for 5200 docs
	df['docs_clean_w_w'] = docs_clean_w_w
	

# Tokenize into clauses
# ============================================================
# 15m samples for 38k CTL convos just texter
# 11m samples for 6500 CTL convos  texter+counselor
	
run_this = False
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

	


	# More cleaning of docs
	# ============================================================
	df['docs_clean_clauses_original'] = df['docs_clean_clauses'].values
	docs_clean_clauses = df['docs_clean_clauses'].values
	docs_clean_clauses_clean = []
	for doc in docs_clean_clauses:
		clauses_doc_i = [] 
		# if doc != []:
		# 	clauses = eval(doc)
		if doc == []:
			docs_clean_clauses_clean.append(doc)
			continue
		for clause in doc:
			clauses_doc_i.extend(clause.split('\n'))
		doc_clean = [n.replace('texter : ','').replace('counselor : ','').replace('\n','. ').strip('.,:\n').replace(" '", "'").replace(' ’', "'").replace(' ,', ',').strip(' ').replace('observer : ', '').replace(" n't", "n't").replace(" n’t", "n't").replace(" ( 1/2 )", "").replace('{ { URL } }', '').replace('[ scrubbed ]','').replace('  ', ' ') for n in clauses_doc_i]
		doc_clean = [n for n in doc_clean if (n not in ['texter', 'counselor', 'observer', '', '-- UNREADABLE MESSAGE --', '( 2/2 )', '( 1/2 )']) and (len(n)>5)]
		
		docs_clean_clauses_clean.append(doc_clean)

	df['docs_clean_clauses']  = docs_clean_clauses_clean
	conversation_ids = df['conversation_id'].values

	print(len(docs_clean_clauses_clean))
	# df.to_csv(f'./data/input/ctl/X_test_all_with_interaction_preprocessed_{ts}.csv')


else:
	docs_clean = [n.replace('\n', '. ') for n in docs_clean] # help tokenize by clause
	df = pd.read_csv('./data/input/ctl/X_test_all_with_interaction_preprocessed_24-03-07T04-25-04.csv')
	





# # Encode embeddings and compute similarity
# ============================================================
# ### Construct (Lexicon)
# ============================================================



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


# prompt_names = dict(zip(ctl_tags13, ['']*len(ctl_tags13)))
prompt_names = {'self_harm': 'self harm or self injury',
 'suicide': 'suicidal thoughts or suicidal behaviors',
 'bully': 'bullying',
 'abuse_physical': 'physical abuse',
 'abuse_sexual': 'sexual abuse',
 'relationship': 'relationship issues',
 'bereavement': 'bereavement or grief',
 'isolated': 'loneliness or social isolation',
 'anxiety': 'anxiety',
 'depressed': 'depression',
 'gender': 'gender identity',
 'eating': 'an eating disorder or body image issues',
 'substance': 'substance use'}

# ============================================================
import dill
import srl_constructs
srl = dill.load(open("./../lexicon/data/input/lexicons/suicide_risk_lexicon_validated_prototypical_tokens_24-03-06T00-47-30.pickle", "rb"))
constructs_to_measure = ctl_tags13.copy()
ctl_tags13_to_srl_name_mapping = srl_constructs.ctl_tags13_to_srl_name_mapping.copy()




word_prototypes = srl_constructs.word_prototypes

# # Encode lexicon
# you want to encode each token once, because can appear in multiple lexicons
# ============================================================

embeddings_dir = './../lexicon/data/input/lexicons/'
prior_embeddings = dill.load(open(embeddings_dir+'embeddings_lexicon-tokens_all-MiniLM-L6-v2.pickle', "rb"))


# For each document exactract CTS constructs
# ============================================================
# For each SRL construct 


# ============================================================ 
run_this = True # wont encode anything if already in prior embeddings

import dill

if run_this:

	tokens_all = []
	for construct in srl.constructs.keys():
		tokens = srl.constructs[construct]['tokens']                      
		tokens_all.extend(tokens)
		
	tokens_all = list(set(tokens_all))


	from sentence_transformers import SentenceTransformer
	embeddings_name = 'all-MiniLM-L6-v2'
	sentence_embedding_model = SentenceTransformer(embeddings_name)       # load embedding
	
	print(sentence_embedding_model .max_seq_length) #default = 256
	tokens_to_encode = [n for n in tokens_all if n not in prior_embeddings.keys()]
			
	if tokens != []:
		embeddings = sentence_embedding_model.encode(tokens_to_encode, convert_to_tensor=True,show_progress_bar=True)	
		embeddings_d = dict(zip(tokens_to_encode, embeddings))
		prior_embeddings.update(embeddings_d)
		# save pickle of embeddings

	with open(embeddings_dir+'embeddings_lexicon-tokens_all-MiniLM-L6-v2.pickle', 'wb') as handle:
		dill.dump(prior_embeddings, handle, protocol=dill.HIGHEST_PROTOCOL)

	# embeddings = pd.DataFrame(embeddings, columns = [f'{embeddings_name}_{str(n).zfill(4)}' for n in range(embeddings.shape[1])])



ctl_tags13_to_srl_name_mapping = srl_constructs.ctl_tags13_to_srl_name_mapping
print(ctl_tags13_to_srl_name_mapping)

construct_prototypes_tokens_d = {}
construct_prototypes_embeddings_d = {}
construct_singletoken_tokens_d = {}
construct_singletoken_embeddings_d = {}

for ctl_construct, srl_constructs in ctl_tags13_to_srl_name_mapping.items():
	# Single tokens
	# print( srl_constructs[0], srl.constructs[srl_constructs[0]]['tokens'])
	
	
	tokens = word_prototypes.get(ctl_construct)
	construct_singletoken_tokens_d[ctl_construct] = tokens
	for token in tokens:
		construct_singletoken_embeddings_d[token] = prior_embeddings[token]
	
	

	# Prototypes (==3 in clinician ratings)
	tokens_i = [] # tokens of srl_constructs 
	for srl_construct in srl_constructs:
		tokens_construct_i = srl.constructs[srl_construct]['tokens']
		tokens_i.extend(tokens_construct_i)
	tokens_i = list(set(tokens_i))
	print(tokens_i)
	if ctl_construct == 'cordial':
		try: tokens_i.remove('cordial')
		except:pass
	construct_prototypes_tokens_d[ctl_construct] = tokens_i
	for token in tokens_i:
		construct_prototypes_embeddings_d[token] = prior_embeddings[token]


# ============================================================


# More cleaning of docs
# ============================================================




docs_clean_clauses = df['docs_clean_clauses'].values

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

# 20 sec
docs_clean_clauses_clean = clean_ctl_conversation(docs_clean_clauses)

df['docs_clean_clauses']  = docs_clean_clauses_clean
conversation_ids = df['conversation_id'].values

print(len(docs_clean_clauses_clean))


# len(embeddings_tokens_docs_d[824727])
# len(df[df['conversation_id']==824727]['docs_clean_clauses'].values[0])



# Encoding docs
# ============================================================
# 100m 6000 with interaction
import pickle

run_this = False 

if run_this:
	embeddings_tokens_docs_d = {}
	for i, (list_of_clauses, conversation_id) in enumerate(zip(docs_clean_clauses_clean, conversation_ids)):
		embeddings_tokens_docs_d[conversation_id] = sentence_embedding_model.encode(list_of_clauses, convert_to_tensor=True,show_progress_bar=False)	# 256
		if i%500==0:
			
			i_str = str(i).zfill(5)
			with open('./data/input/ctl/embeddings/'+f'embeddings_{embeddings_name}_docs_clauses_with-interaction_{ts}_part-{i_str}.pickle', 'wb') as handle:
				pickle.dump(embeddings_tokens_docs_d, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('./data/input/ctl/embeddings/'+f'embeddings_{embeddings_name}_docs_clauses_with-interaction_{ts}.pickle', 'wb') as handle:
		pickle.dump(embeddings_tokens_docs_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
	# load pickle	
	with open('./data/input/ctl/embeddings/'+f'embeddings_all-MiniLM-L6-v2_docs_clauses_with-interaction_24-03-07T04-25-04.pickle', 'rb') as handle:
		embeddings_tokens_docs_d = pickle.load(handle)





len(embeddings_tokens_docs_d.keys())


# CTS using both types of construct representation (1 token and multiple) and for content validity test sets
# ============================================================
# ============================================================

from importlib import reload
reload(cts)

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
from sklearn.preprocessing import MinMaxScaler


import numpy as np





run_this = False 

if run_this:
	# Lexicon single token
	# ============================================================
	# computing similarity between 13 constructs and 13842 documents --> 1.40m
	method = 'lexicon_clause' # single token - clauses
	feature_vectors_i_wc, cosine_scores_docs_i_wc = cts.measure(
				construct_tokens_d = construct_singletoken_tokens_d,
				construct_embeddings_d = construct_singletoken_embeddings_d,
				# docs = docs_clean_clauses,
				docs_embeddings_d = embeddings_tokens_docs_d,
				method = method, #todo: change to token, tokens, weighted_tokens
				summary_stat = ['max'],
				return_cosine_similarity=True,
				minmaxscaler = (0,1) # scaling all values between 0 and 1. 
			)

	# TODO: Plot histogram for both groups before and after
	feature_vectors_i_wc.describe()
	dfs['test']['cts_token_clause'] = feature_vectors_i_wc.copy()


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



	# lexicon_protoypes
	# ============================================================
	from importlib import reload
	reload(cts)


	run_this = False

	if run_this:
		method = 'lexicon_clause'
		feature_vectors_lc, cosine_scores_docs_lc = cts.measure(
					construct_tokens_d = construct_prototypes_tokens_d,
					construct_embeddings_d = construct_prototypes_embeddings_d,
					# docs = docs_clean_clauses,
					docs_embeddings_d = embeddings_tokens_docs_d,
					method = method, #todo: change to token, tokens, weighted_tokens
					summary_stat = ['max'],
					return_cosine_similarity=True,
					minmaxscaler = (0,1)
				)

		# TODO: Plot histogram for both groups before and after
		feature_vectors_lc.describe()
		dfs['test']['cts_prototypes_clause'] = feature_vectors_lc.copy()

		feature_vectors_lc.to_csv('./data/input/ctl/X_test_all_with_interaction_cts_prototypes_clause.csv')

		cosine_scores_docs_lc.keys()
		# save dictionary using dill .pickle
		import dill
		dill.dump(cosine_scores_docs_lc, open('./data/input/ctl/X_test_all_with_interaction_cts_prototypes_clause_cosine_scores.dill', 'wb'))


	# Save 
	run_this = True #True saves, False loads
	if run_this:
		# Save
		with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'wb') as f:
			pickle.dump(dfs, f) 
	else:
		# Load
		with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'rb') as f:
			dfs = pickle.load(f)



	# content validity test sets: extract features on these tokens
	# ===================================

	for split in ['X_test_content_validity_prototypicality-3']:
	# for split in ['X_test_content_validity_prototypicality-1_3', 'X_test_content_validity_prototypicality-3']:
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
		feature_vectors_i_lc_content, cosine_scores_docs_i_lc_content = cts.measure(
				construct_tokens_d = construct_prototypes_tokens_d,
				construct_embeddings_d = construct_prototypes_embeddings_d,
				# docs = docs_clean_clauses,
				docs_embeddings_d = embeddings_tokens_docs_content_validity_d,
				method = method, #todo: change to token, tokens, weighted_tokens
				summary_stat = ['max'],
				return_cosine_similarity=True,
				minmaxscaler = (0,1)
			)
		
		if df_i.shape[0]==feature_vectors_i_lc_content.shape[0]:
			df_i[feature_vectors_i_lc_content.columns] = feature_vectors_i_lc_content.values
		
		dfs[split]['cts_prototypes_clause'] = df_i.copy()








	if run_this:
		# Save
		with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'wb') as f:
			pickle.dump(dfs, f) 





# Evaluate on test set
# ============================================================
from sklearn.metrics import roc_curve
import numpy as np


# Load
with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'rb') as f:
	dfs = pickle.load(f)

# Example usage




# 64,51,54 vs .4, .25, 56 (with much more training data)
np.random.seed(123)

# TODO: see where to save feature_vector (tfidf, liwc22) and where to save model_name (Ridge, LightGBM)
import dill
def load_lexicon(path):
	lexicon = dill.load(open(path, "rb"))
	return lexicon
srl = load_lexicon("./../lexicon/data/input/lexicons/suicide_risk_lexicon_validated_24-03-06T00-37-15.pickle")
constructs_in_order = list(srl.constructs.keys())


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
	'cts_token_clause',

	'cts_prototypes_clause' 
				   ] 


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


ts_i = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
results = []
results_content_validity = []
# for gridsearch in [True]:
from concept_tracker.utils import metrics_report
from sklearn import metrics

best_params = None

n = 0
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
		y_proba_1 = [n[0] for n in y_proba_1]
		
		optimal_threshold = np.round(metrics_report.find_optimal_threshold(y_test, y_proba_1),3)
		threshold_05 = 0.5

		
		
		# content validity test sets. y_proba_1_3_dv = y_proba_1 for prototypicality==3
		y_proba_1_3_dv = X_test_3[X_test_3['y_test']==dv][feature_names].values
		y_proba_1_3_dv = [n[0] for n in y_proba_1_3_dv]
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









