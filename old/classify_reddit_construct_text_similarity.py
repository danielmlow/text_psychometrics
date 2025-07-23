#!/usr/bin/env python
# coding: utf-8

"""
1. Obtain CTS scores using two methods: 1 prototype and multiple prototypes
"""
# !pip install -U spacy
# !python -m spacy download en_core_web_sm

import pandas as pd
import numpy as np
import os
import datetime
import pickle
import seaborn as sns
from construct_tracker import cts
from construct_tracker.utils import stop_words, clean, lemmatizer, stop_words, tokenizer
from construct_tracker import lexicon
from importlib import reload
reload(cts)


# ============================================================
input_dir = './data/input/reddit/'
output_dir = './data/'

task = 'classification'
ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S') # for saving outputs

## Load data
# ============================================================

# load pickle
with open('./data/input/reddit/reddit_13_mental_health_4600_posts_20250311_123431_dfs.pkl', 'rb') as handle:
	dfs = pickle.load(handle)


srl = lexicon.load_lexicon(name = 'srl_v1-0') # Load lexicon
srl_prototypes = lexicon.load_lexicon(name = 'srl_prototypes_v1-0') # Load lexicon
constructs_in_order = list(srl.constructs.keys())



# TODO: add features and save to pickle
# dfs['test'].update(
# 	{"ctl":X_test}
# )


# config
# ============================================================
construct_representation = 'lexicon'
document_representation = 'clause'
embedding_type = 'sentence'
embedding_model = 'all-MiniLM-L6-v2'

# Tokenize into clauses
# ============================================================


run_this = False
if run_this:
	X_test = dfs['test']['srl_validated'].copy()
	X_test = X_test.drop(constructs_in_order, axis=1)


	
	X_test.to_csv(f'./data/input/reddit/construct_text_similarity_X_test_clean_{ts}.csv')


else:
	X_test = pd.read_csv(f'./data/input/reddit/construct_text_similarity_X_test_clean_25-03-18T20-09-28.csv')
	
	

# # Encode embeddings and compute similarity
# ============================================================
# ### Construct (Lexicon)
# ============================================================

# Subreddits for each construct
# dv_tags = {
# 	'self_harm': 'selfharm',
#  'suicide': 'SuicideWatch',
#  'bully': 'bullying',
#  'abuse_sexual': 'sexualassault',
#  'bereavement': 'GriefSupport',
#  'isolated': 'lonely',
#  'anxiety': 'Anxiety',
#  'depressed': 'depression',
#  'gender': 'AskLGBT',
#  'eating': 'EatingDisorders',
#  'substance': 'addiction',
#  }


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

# srl_mapping =  {
 
#  'Direct self-injury': 'self_harm',
#  'Active suicidal ideation & suicidal planning': 'suicide',
#  'Passive suicidal ideation': 'suicide',
#  'Other suicidal language': 'suicide',
#  'Bullying': 'bully',
#  'Sexual abuse & harassment': 'abuse_sexual',
#  'Grief & bereavement': 'bereavement',
#  'Loneliness & isolation': 'isolated',
#  'Anxiety': 'anxiety',
#  'Depressed mood': 'depressed',
#  'Gender & sexual identity': 'gender',
#  'Eating disorders': 'eating',
#  'Other substance use': 'substance',
#  'Alcohol use':'substance',

#  }

# srl_mapping_r = {v: k for k, v in srl_mapping.items()}

srl_mapping = {'self_harm': 'Direct self-injury',
 'suicide': ['Active suicidal ideation & suicidal planning','Passive suicidal ideation','Other suicidal language'],
 'bully': 'Bullying',
 'abuse_sexual': 'Sexual abuse & harassment',
 'bereavement': 'Grief & bereavement',
 'isolated': 'Loneliness & isolation',
 'anxiety': 'Anxiety',
 'depressed': 'Depressed mood',
 'gender': 'Gender & sexual identity',
 'eating': 'Eating disorders',
 'substance': ['Other substance use','Alcohol use']}


single_prototype = {
    'self_harm': ['self harm or self injury'],
    'suicide': ['suicidal thoughts or suicidal behaviors'],
    'bully': ['bullying'],
    'abuse_sexual': ['sexual abuse'],
    'bereavement': ['bereavement or grief'],
    'isolated': ['loneliness or social isolation'],
    'anxiety': ['anxiety'],
    'depressed': ['depression'],
    'gender': ['gender identity'],
    'eating': ['an eating disorder or body image issues'],
    'substance': ['substance use']
}


multiple_prototypes = {}
for k,v in srl_mapping.items():
	if type(v) == str:
		multiple_prototypes[k] = srl_prototypes.constructs[v]['tokens'].copy()
	else:
		multiple_prototypes[k] = []
		for construct in v:
			multiple_prototypes[k].extend(srl_prototypes.constructs[construct]['tokens'])


multiple_prototypes['bully']


# Extract CTS
# ============================================================
import time

documents = X_test['document'].values
# Clean
documents = [n.replace('\n---\n', '. ').replace('. .', '.').replace('?.', '?').replace('? .', '?').replace('!.', '!').replace('! .', '!').replace('....', '...').replace('...', '... ').strip('.').replace('\n', ' ').replace('  ', ' ').replace(" n't", "n't").replace(" 's", "'s").replace(" ’s", "'s").replace(" ’m", "'m").replace(' nt', 'nt').replace(" n’t", "n't").replace(" n’t", "n't").strip(' ').replace(' . ', '. ').replace(' : ', ': ').replace('.,', '.') for n in documents]

feature_vectors_lc_all = []
cosine_scores_docs_lc_all = []
# 4m samples for 7800 reddit posts
for lexicon_dict in [single_prototype, multiple_prototypes]:
	start = time.time()
	feature_vectors_lc, cosine_scores_docs_lc = cts.measure(lexicon_dict,
			documents,
			construct_representation = construct_representation,
			document_representation = document_representation,
			count_if_exact_match = False,
			)
	feature_vectors_lc_all.append(feature_vectors_lc)
	cosine_scores_docs_lc_all.append(cosine_scores_docs_lc)
	end = time.time()
	print(end - start)



# Extract for content validity test sets
X_test_3 = dfs['content_validity']['srl_validated'].copy()

for lexicon_dict in [single_prototype, multiple_prototypes]:
	start = time.time()
	feature_vectors_lc, cosine_scores_docs_lc = cts.measure(lexicon_dict,
			documents,
			construct_representation = construct_representation,
			document_representation = document_representation,
			count_if_exact_match = False,
			)
	feature_vectors_lc_all.append(feature_vectors_lc)
	cosine_scores_docs_lc_all.append(cosine_scores_docs_lc)
	end = time.time()
	print(end - start)




# Save 
# ===================================================
os.makedirs('./data/input/reddit/construct_text_similarity_features/', exist_ok=True)
for df_i, cosines, name in zip(feature_vectors_lc_all,cosine_scores_docs_lc_all, ['single', 'multi']):
	display(df_i)
	df_i.to_csv(f'./data/input/reddit/construct_text_similarity_features/construct_text_similarity_feature_vectors_lc_{name}.csv')
	# save dict as pickle
	with open(f'./data/input/reddit/construct_text_similarity_features/construct_text_similarity_cosine_scores_docs_lc_{name}.pkl', 'wb') as f:
		pickle.dump(cosines, f)
	
# TODO: test if correct

X_test_merged = pd.concat([X_test, feature_vectors_lc_all[0].rename(columns = {'document': f'document_clean'})], axis=1)
X_test_merged[['title', 'document', 'document_clean']].sample(10)
dfs['test'].update({'ctl_single': X_test_merged})

X_test_merged = pd.concat([X_test, feature_vectors_lc_all[1].rename(columns = {'document': f'document_clean'})], axis=1)
X_test_merged[['title', 'document', 'document_clean']].sample(10)
dfs['test'].update({'ctl_multi': X_test_merged})

# Save dfs as pickle
with open('./data/input/reddit/reddit_13_mental_health_4600_posts_20250311_123431_dfs.pkl', 'wb') as handle:
	pickle.dump(dfs, handle)




