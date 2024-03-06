#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!pip install -q pyarrow


# # Load dataset



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
import sys
sys.path.append( './../../concept-tracker/')
# from concept_tracker import lexicon

ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')

pd.set_option("display.max_columns", None)
# pd.options.display.width = 0


# os.chdir(os.path.dirname(__file__)) # Set working directory to current file

on_colab = False
location = 'local'


if location == 'openmind':
  input_dir = '/nese/mit/group/sig/projects/dlow/ctl/datasets/'
  output_dir = 'home/dlow/zero_shot/data/output/'
elif location =='local':
  input_dir = '/Users/danielmlow/data/ctl/input/datasets/'
  output_dir = './data/output/'
os.makedirs(output_dir, exist_ok=True)

train = pd.read_parquet(input_dir + f'train10_train_metadata_messages_clean.gzip', engine='pyarrow')
test = pd.read_parquet(input_dir + f'train10_test_metadata_messages_clean.gzip', engine='pyarrow')





# Config
# ====================================================================================
task = 'classification'
normalize_lexicon = True
with_interaction = True
n_per_dv = 300




# In[271]:


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






if with_interaction:
	max_length = int(1750*1.4)+75 #word count * 1.4 +75 for the prompt ~ tokens, 98%have less than this
else:
	# just texter     
	max_length = int(1000*1.4)+75

# Balance dataset:
# ====================================================================================================================================
def create_binary_dataset(df_metadata, dv = 'suicide', n_per_dv = 3000):
	df_metadata_tag_1 = df_metadata[df_metadata[dv]==1].sample(n=n_per_dv,random_state=123)
	df_metadata_tag_0 = df_metadata[df_metadata[dv]==0].sample(n=n_per_dv,random_state=123)
	assert df_metadata_tag_1.shape[0] == n_per_dv
	assert df_metadata_tag_0.shape[0] == n_per_dv

	df_metadata_tag = pd.concat([df_metadata_tag_1, df_metadata_tag_0]).sample(frac=1).reset_index(drop=True)

	return df_metadata_tag

if run_this:
	n_per_dvs = [50,150,2000]

	X_train_all = []
	X_test_all = []

	for n_per_dv in n_per_dvs:

		for dv in ctl_tags13:
			# output_dir_i_dv = output_dir_i+f'{dv}/'
			# os.makedirs(output_dir_i_dv, exist_ok = True)
				

			responses = []
			time_elapsed_all = []
			# add_file_handler(output_dir_i_dv+f'log_print_statements_{dv}.txt')
		#     print = custom_print
			
			# construct = prompt_names.get(dv)
			# Configure logging

			# Test set
			train_i = create_binary_dataset(train, dv = dv, n_per_dv = n_per_dv) #random_state = 123
			test_i = create_binary_dataset(test, dv = dv, n_per_dv = 300)#random_state = 123
			
			
			if with_interaction:
				X_train_df = train_i[['conversation_id','message_with_interaction_clean', dv]]
				X_test_df = test_i[['conversation_id','message_with_interaction_clean', dv]]
				
			else:
				X_train_df = train_i[['conversation_id','message_with_clean', dv]]
				X_test_df = test_i[['conversation_id','message_with_clean', dv]]
			# y_train = train_i[dv].values
			# y_test = test_i[dv].values
			X_train_all.append(X_train_df)
			X_test_all.append(X_test_df)


			# print('\n', dv, '============================================')
			# print(test_i[ctl_tags13].sum())
			# print('construct:', construct)
			# print('len of documents:',len(X_test))
			# print('len of y_test:',len(y_test))

	X_train_all
	X_train_all = pd.concat(X_train_all)
	X_test_all = pd.concat(X_test_all)

	print(X_train_all.shape)
	# remove duplicates
	X_train_all.drop_duplicates(subset=['conversation_id'],inplace=True)
	X_test_all.drop_duplicates(subset=['conversation_id'],inplace=True)
	print(X_train_all.shape)

	X_train_all.to_csv('./data/input/ctl/X_train_all_with_interaction.csv', index = False)
	X_test_all.to_csv('./data/input/ctl/X_test_all_with_interaction.csv', index = False)

	dfs = {}
	dfs['train'] = {}
	dfs['test'] = {}
	dfs['train']['df_text'] = X_train_all.rename(columns = {'message_with_interaction_clean': 'text'}).copy()
	dfs['test']['df_text'] = X_test_all.rename(columns = {'message_with_interaction_clean': 'text'}).copy()

import pickle
run_this = True #True saves, False loads
if run_this:
    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'wb') as f:
        pickle.dump(dfs, f) 
else:

    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'rb') as f:
    	dfs = pickle.load(f)





# Create content validity test sets
# ================================================================================================
import dill			
			
def load_lexicon(path):
	lexicon = dill.load(open(path, "rb"))
	return lexicon
srl = load_lexicon("./../lexicon/data/input/lexicons/suicide_risk_lexicon_validated_24-03-06T00-37-15.pickle")

ctl_tags13_to_srl_name_mapping = {'self_harm': ['Direct self-injury'],
 'suicide': ['Active suicidal ideation & suicidal planning',
  'Passive suicidal ideation',
  'Other suicidal language'],
 'bully': ['Bullying'],
 'abuse_physical': ['Physical abuse & violence'],
 'abuse_sexual': ['Sexual abuse & harassment'],
 'relationship': ['Relationship issues'],
 'bereavement': ['Grief & bereavement'],
 'isolated': ['Loneliness & isolation'],
 'anxiety': ['Anxiety'],
 'depressed': ['Depressed mood'],
 'gender': ['Gender & sexual identity'],
 'eating': ['Eating disorders'],
 'substance': ['Other substance use', 'Alcohol use']}


X_test_content_validity = []
for dv in ctl_tags13:
	constructs = ctl_tags13_to_srl_name_mapping.get(dv)
	# get tokens from srl	
	tokens_all = []
	for construct in constructs: 
		
		tokens_all.extend(srl.constructs[construct]['tokens'])

	X_test_content_validity_i = pd.DataFrame({
		'token':tokens_all,
		'y_test':[dv]*len(tokens_all),
		'constructs':['--'.join(constructs)]*len(tokens_all)
	})
	X_test_content_validity.append(X_test_content_validity_i)

X_test_content_validity = pd.concat(X_test_content_validity)





#Extract LIWC
# ====================================================================================================================


# # Skip loading data and extracting featues and load below

# # Or load data and extract




# Suicide risk lexicon. should be able to import it




# # Descriptive statistics

# dataset_name = 'train10_subset_30'

# for split in dfs.keys():
# 	df_text = dfs[split]['df_text'][['conversation_id', 'text', 'y']]
# 	df_text.to_csv(f'./data/input/ctl/{dataset_name}_{split}_text_y.csv', index = False)


# # LIWC

# In[283]:


dv_distr


# In[284]:


# # Remove non IV columns from LIWC22
# for split in dfs.keys():
#     dfs[split]['liwc22'] = dfs[split]['liwc22'].drop(['Segment', 'conversation_id', 'message', 'Emoji'], axis=1)
#     if balance and split=='train':
#         dfs[split]['liwc22_balanced'] = dfs[split]['liwc22'].drop(['Segment', 'conversation_id', 'message', 'Emoji'], axis=1)
                    


# # Extract liwc

# ## automated liwc

# In[285]:


# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # Ryan L. Boyd
# # 2022-03-17


# run_this = False # We'll do it manually below
# 	if run_this:


# 		# This is an example script that demonstrates how to make a call to the LIWC-22 command line interface (CLI)
# 		# from Python. Briefly described, what we want to do is launch the CLI application as a subprocess, then wait
# 		# for that subprocess to finish.

# 		# This is a very crude example script, so please feel free to improve/innovate on this example :)          """


# 	# Make sure that you have the LIWC-22.exe GUI running — it is required for the CLI to function correctly :)
# 	# Make sure that you have the LIWC-22.exe GUI running — it is required for the CLI to function correctly :)
# 	# Make sure that you have the LIWC-22.exe GUI running — it is required for the CLI to function correctly :)
# 	# Make sure that you have the LIWC-22.exe GUI running — it is required for the CLI to function correctly :)


# 	import subprocess


# 	#  ______    _     _                      _ _   _       _________   _________   ______ _ _
# 	# |  ____|  | |   | |                    (_| | | |     |__   __\ \ / |__   __| |  ____(_| |
# 	# | |__ ___ | | __| | ___ _ __  __      ___| |_| |__      | |   \ V /   | |    | |__   _| | ___ ___
# 	# |  __/ _ \| |/ _` |/ _ | '__| \ \ /\ / | | __| '_ \     | |    > <    | |    |  __| | | |/ _ / __|
# 	# | | | (_) | | (_| |  __| |     \ V  V /| | |_| | | |    | |   / . \   | |    | |    | | |  __\__ \
# 	# |_|  \___/|_|\__,_|\___|_|      \_/\_/ |_|\__|_| |_|    |_|  /_/ \_\  |_|    |_|    |_|_|\___|___/

# 	inputFolderTXT = "C:/Users/Ryan/Datasets/TED - English Only - TXT Files/"
# 	outputLocation = "C:/Users/Ryan/Datasets/TED Talk TXT Files - Analyzed.csv"

# 	# This command will read texts from a folder, analyze them using the standard "Word Count" LIWC analysis,
# 	# then save our output to a specified location.
# 	cmd_to_execute = ["LIWC-22-cli",
# 					"--mode", "wc",
# 					"--input", inputFolderTXT,
# 					"--output", outputLocation]



# 	# Let's go ahead and run this analysis:
# 	subprocess.call(cmd_to_execute)

# 	# We will see the following in the terminal as it begins working:
# 	#
# 	#    Picked up JAVA_TOOL_OPTIONS: -Dfile.encoding=UTF-8
# 	#    Processing:
# 	#     - [folder] C:\Users\Ryan\Datasets\TED - English Only - TXT Files
# 	#    [===================                     ] 47.75%; Number of Texts Analyzed: 1304; Total Words Analyzed: 2.62M


# 	# A thing of beauty, to be sure. What if we want to process our texts using an older LIWC dictionary,
# 	# or an external dictionary file? This can be done easily as well.



# 	# We can specify whether we want to use the LIWC2001, LIWC2007, LIWC2015,
# 	# or LIWC22 dictionary with the --dictionary argument.
# 	liwcDict = "LIWC2015"

# 	# Alternatively, you can specify the absolute path to an external dictionary
# 	# file that you would like to use, and LIWC will load this dictionary for processing.
# 	#liwcDict = "C:/Users/Ryan/Dictionaries/Personal Values Dictionary.dicx"


# 	# Let's update our output location as well so that we don't overwrite our previous file.
# 	outputLocation = "C:/Users/Ryan/Datasets/TED Talk TXT Files - Analyzed (LIWC2015).csv"

# 	cmd_to_execute = ["LIWC-22-cli",
# 					"--mode", "wc",
# 					"--dictionary", liwcDict,
# 					"--input", inputFolderTXT,
# 					"--output", outputLocation]

# 	subprocess.call(cmd_to_execute)








# 	#   _____  _______      __  ______ _ _
# 	#  / ____|/ ____\ \    / / |  ____(_| |
# 	# | |    | (___  \ \  / /  | |__   _| | ___
# 	# | |     \___ \  \ \/ /   |  __| | | |/ _ \
# 	# | |____ ____) |  \  /    | |    | | |  __/
# 	#  \_____|_____/    \/     |_|    |_|_|\___|



# 	# Beautiful. Now, let's do the same thing, but analyzing a CSV file full of the same texts.
# 	inputFileCSV = 'C:/Users/Ryan/Datasets/TED Talk - English Transcripts.csv'
# 	outputLocation = 'C:/Users/Ryan/Datasets/TED Talk CSV File - Analyzed.csv'


# 	# We're going to use a variation on the command above. Since this is a CSV file, we want to include the indices of
# 	#     1) the columns that include the text identifiers (although this is not required, it makes our data easier to merge later)
# 	#     2) the columns that include the actual text that we want to analyze
# 	#
# 	# In my CSV file, the first column has the text identifiers, and the second column contains the text.
# 	# For more complex datasets, please use the --help argument with LIWC-22 to learn more about how to process your text.
# 	cmd_to_execute = ["LIWC-22-cli",
# 					"--mode", "wc",
# 					"--input", inputFileCSV,
# 					"--row-id-indices", "1",
# 					"--column-indices", "2",
# 					"--output", outputLocation]

# 	# Let's go ahead and run this analysis:
# 	subprocess.call(cmd_to_execute)


# 	# We will see the following in the terminal as LIWC does its magic:
# 	#    Picked up JAVA_TOOL_OPTIONS: -Dfile.encoding=UTF-8
# 	#    Processing:
# 	#     - [file] C:\Users\Ryan\Datasets\TED Talk - English Transcripts.csv
# 	#    [========================================] 100.00%; Number of Rows Analyzed: 2737; Total Words Analyzed: 5.40M
# 	#    Done. Please examine results in C:\Users\Ryan\Datasets\TED Talk CSV File - Analyzed.csv









# 	#                       _                  _____ _        _
# 	#     /\               | |                / ____| |      (_)
# 	#    /  \   _ __   __ _| |_   _ _______  | (___ | |_ _ __ _ _ __   __ _
# 	#   / /\ \ | '_ \ / _` | | | | |_  / _ \  \___ \| __| '__| | '_ \ / _` |
# 	#  / ____ \| | | | (_| | | |_| |/ |  __/  ____) | |_| |  | | | | | (_| |
# 	# /_/    \_|_| |_|\__,_|_|\__, /___\___| |_____/ \__|_|  |_|_| |_|\__, |
# 	#                          __/ |                                   __/ |
# 	#                         |___/                                   |___/

# 	# What if we want to simply pass a string to the CLI for analysis? This is possible. As described on the
# 	# Help section of the liwc.app website, this is generally not recommended as it will not be very performant.
# 	#
# 	# Also, of serious importance! Most command lines/terminals have a limit on the length of any string that it
# 	# will parse. This means that you likely cannot analyze very long texts (e.g., like a long paper, speech,
# 	# or book) by passing the text directly into the console. Instead, you will likely need to process your
# 	# data directly from the disk instead.
# 	#
# 	# However, if you insist...

# 	# The string that we would like to analyze.
# 	inputString = "This is some text that I would like to analyze. After it has finished, I will say \"Thank you, LIWC!\""

# 	# For this one, let's save our result as a newline-delimited json file (.ndjson)
# 	outputLocation = 'C:/Users/Ryan/Datasets/LIWC-22 Results from String.ndjson'


# 	cmd_to_execute = ["LIWC-22-cli",
# 					"--mode", "wc",
# 					"--input", "console",
# 					"--console-text", inputString,
# 					"--output", outputLocation]


# 	# Let's go ahead and run this analysis:
# 	subprocess.call(cmd_to_execute)

# 	# The results from this analysis:
# 	#{"Segment": 1,"WC": 20,"Analytic": 3.8,"Clout": 40.06,"Authentic": 28.56,"Tone": 99,"WPS": 10,"BigWords": 10,
# 	#"Dic": 100, "Linguistic": 80,"function": 70,"pronoun": 30,"ppron": 15,"i": 10,"we": 0,"you": 5,"shehe": 0,"they": 0,
# 	#"ipron": 15,"det": 15,"article": 0,"number": 0,"prep": 15,"auxverb": 20,"adverb": 0,"conj": 5,"negate": 0,
# 	#"verb": 35,"adj": 0,"quantity": 5,"Drives": 5,"affiliation": 0,"achieve": 5,"power": 0,"Cognition": 15,
# 	#"allnone": 0,"cogproc": 15,"insight": 5,"cause": 0,"discrep": 10,"tentat": 0,"certitude": 0,"differ": 0,
# 	#"memory": 0,"Affect": 15,"tone_pos": 15,"tone_neg": 0,"emotion": 10,"emo_pos": 10,"emo_neg": 0,"emo_anx": 0,
# 	#"emo_anger": 0,"emo_sad": 0,"swear": 0,"Social": 20,"socbehav": 15,"prosocial": 5,"polite": 5,"conflict": 0,"moral": 0,
# 	#"comm": 15,"socrefs": 5,"family": 0,"friend": 0,"female": 0,"male": 0,"Culture": 5,"politic": 0,"ethnicity": 0,"
# 	#tech": 5,"Lifestyle": 0,"leisure": 0,"home": 0,"work": 0,"money": 0,"relig": 0,"Physical": 0,"health": 0,"illness": 0,
# 	#"wellness": 0,"mental": 0,"substances": 0,"sexual": 0,"food": 0,"death": 0,"need": 0,"want": 0,"acquire": 0,"lack": 0,
# 	#"fulfill": 0,"fatigue": 0,"reward": 0,"risk": 0,"curiosity": 0,"allure": 0,"Perception": 0,"attention": 0,"motion": 0,
# 	#"space": 0,"visual": 0,"auditory": 0,"feeling": 0,"time": 10,"focuspast": 0,"focuspresent": 10,"focusfuture": 5,
# 	#"Conversation": 0,"netspeak": 0,"assent": 0,"nonflu": 0,"filler": 0,
# 	#"AllPunc": 30,"Period": 5,"Comma": 10,"QMark": 0,"Exclam": 5,"Apostro": 0,"OtherP": 10}



# 	# And, lastly — what if we want to get the output directly from the command line or terminal as a json string?
# 	# Why, we can do that too!


# 	inputString = "This is some text that I would like to analyze. After it has finished," \
# 				" we will get results in the console. Hooray!"
# 	outputLocation = "console"

# 	cmd_to_execute = ["LIWC-22-cli",
# 					"--mode", "wc",
# 					"--input", "console",
# 					"--console-text", inputString,
# 					"--output", outputLocation]

# 	# Let's go ahead and run this analysis. We do this somewhat differently than what we've been doing, however.
# 	# This will end up giving us a list, where each element is a line of output from the console.
# 	results = subprocess.check_output(cmd_to_execute, shell=True).strip().splitlines()

# 	# In this case, the item that we want to parse from a json to a Python dictionary is in results[1], so we will
# 	# go right ahead and parse that to a dictionary now:
# 	import json
# 	results_json = json.loads(results[1])


# ## Manual

# In[286]:




# In[287]:


dfs[split]['df_text']


# In[293]:


liwc_dir = './data/input/ctl/'


liwc_train = pd.read_csv(liwc_dir+f'X_train_all_with_interaction_liwc22.csv')
liwc_test = pd.read_csv(liwc_dir+f'X_test_all_with_interaction_liwc22.csv')


split = 'train'
for split, df_i in zip(['train', 'test'], [liwc_train, liwc_test]):
	

	df_text = dfs[split]['df_text'].copy()
	df_i = df_i[df_i['conversation_id'].isin(df_text['conversation_id'].unique())]
	dfs[split]['liwc22'] = df_i.drop(['Segment', 'conversation_id', 'y', 'text', 'Emoji'], axis=1)
	dfs[split]['liwc22'] = df_i['y'].values
                                   


# # Extract Suicide Risk Lexicon

# In[24]:




# In[26]:



	


# In[296]:


import dill
from concept_tracker.utils import lemmatizer # local script
import tqdm
from concept_tracker.lexicon import lemmatize_tokens
sys.path.append( './../../concept-tracker/') # TODO: replace with pip install construct-tracker
from concept_tracker import lexicon





def load_lexicon(path):
	lexicon = dill.load(open(path, "rb"))
	return lexicon
srl = load_lexicon("./../lexicon/data/input/lexicons/suicide_risk_lexicon_validated_24-03-06T00-37-15.pickle")

srl.exact_match_n

for split in tqdm.tqdm(['train', 'test']):
	
	print('extracting', split)
	df_text = dfs[split]['df_text']
	docs = df_text['text'].values
	
	# for construct in srl.constructs.keys():
	# 	srl.constructs[construct]['remove'] = []

	srl = lemmatize_tokens(srl) 

	# Extract
	feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(docs,
																						srl.constructs,normalize = True, return_matches=True,
																						add_lemmatized_lexicon=True, lemmatize_docs=False,
																						exact_match_n = srl.exact_match_n,exact_match_tokens = srl.exact_match_tokens)
	

	

	df_text[feature_vectors.columns] = feature_vectors.values
	dfs[split]['srl_validated'] = df_text.copy()
	



dfs['train']['srl_validated'].to_csv('./data/input/ctl/X_train_all_with_interaction_srl_validated.csv', index = False)
dfs['test']['srl_validated'].to_csv('./data/input/ctl/X_test_all_with_interaction_srl_validated.csv', index = False)


# # # Suicide Risk Lexicon (only GPT-4 Turbo tokens) 

# # In[304]:


# srl_gpt4 = {}

# for construct in srl.constructs.keys():
#     gpt4_tokens = []
#     for source in srl.constructs[construct]['tokens_metadata'].keys():
#         if 'gpt-4-1106-preview' in source:
#             tokens_i = srl.constructs[construct]['tokens_metadata'][source]['tokens']
#             gpt4_tokens.extend(tokens_i)
            
            
#     srl_gpt4[construct]={'tokens':list(np.unique(gpt4_tokens))}
    


# # In[305]:


# # we'll consider the 2 version of two of these after editing either the definition or prompt_name
# srl_gpt4['Direct self-injury'] = srl_gpt4['Direct self-injury 2'].copy()
# del srl_gpt4['Direct self-injury 2']
# srl_gpt4['Relationship issues'] = srl_gpt4['Relationship issues 2'].copy()
# del srl_gpt4['Relationship issues 2']


# # In[308]:


# list(srl_gpt4.keys())


# # In[309]:


# from concept_tracker.utils import lemmatizer
# for c in tqdm.tqdm(list(srl_gpt4.keys())):
# 	lexicon_tokens = srl_gpt4[c]['tokens']


# 	# If you add lemmatized and nonlemmatized you'll get double count in many cases ("plans" in doc will be matched by "plan" and "plans" in lexicon)
# 	lexicon_tokens_lemmatized = lemmatizer.spacy_lemmatizer(lexicon_tokens, language='en') # custom function
# 	lexicon_tokens_lemmatized = [' '.join(n) for n in lexicon_tokens_lemmatized]
# 	lexicon_tokens += lexicon_tokens_lemmatized
# 	lexicon_tokens = list(np.unique(lexicon_tokens)) # unique set
# 	srl_gpt4[c]['tokens_lemmatized']=lexicon_tokens


# # In[311]:


# for split in ['train', 'test']:
#     df_text = dfs[split]['df_text']
#     docs = df_text['text'].values    
#     feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(docs,
#                                                                                           srl_gpt4,
#                                                                                           normalize = normalize_lexicon,
#                                                                                           exact_match_n = srl.exact_match_n
#                                                                                           )
#     df_text[feature_vectors.columns] = feature_vectors.values
#     dfs[split]['SRL GPT-4 Turbo'] = df_text.drop(['conversation_id', 'y', 'text', 'word_count'], axis=1).copy()

    


# # TextDescriptives
# ========================================================================================================================

# !pip install textdescriptives==2.7.3


td_columns = ['token_length_mean',
#  'token_length_median',
 'token_length_std',
 'sentence_length_mean',
#  'sentence_length_median',
 'sentence_length_std',
#  'syllables_per_token_mean',
#  'syllables_per_token_median',
#  'syllables_per_token_std',
 'n_tokens',
#  'n_unique_tokens',
#  'proportion_unique_tokens',
#  'n_characters',
 'n_sentences',
#  'first_order_coherence',
#  'second_order_coherence',
 'pos_prop_ADJ',
 'pos_prop_ADP',
 'pos_prop_ADV',
 'pos_prop_AUX',
 'pos_prop_CCONJ',
 'pos_prop_DET',
 'pos_prop_INTJ',
 'pos_prop_NOUN',
 'pos_prop_NUM',
 'pos_prop_PART',
 'pos_prop_PRON',
 'pos_prop_PROPN',
 'pos_prop_PUNCT',
 'pos_prop_SCONJ',
 'pos_prop_SYM',
 'pos_prop_VERB',
 'pos_prop_X',
#  'flesch_reading_ease',
#  'flesch_kincaid_grade',
#  'smog',
 'gunning_fog',
 'automated_readability_index',
#  'coleman_liau_index',
#  'lix',
#  'rix',
#  'entropy',
#  'perplexity',
#  'per_word_perplexity',
 'passed_quality_check',
#  'n_stop_words',
 'alpha_ratio',
 'mean_word_length',
#  'doc_length',
 'symbol_to_word_ratio_#',
 'proportion_ellipsis',
#  'proportion_bullet_points',
#  'contains_lorem ipsum',
#  'duplicate_line_chr_fraction',
#  'duplicate_paragraph_chr_fraction',
#  'duplicate_ngram_chr_fraction_5',
#  'duplicate_ngram_chr_fraction_6',
#  'duplicate_ngram_chr_fraction_7',
#  'duplicate_ngram_chr_fraction_8',
#  'duplicate_ngram_chr_fraction_9',
#  'duplicate_ngram_chr_fraction_10',
 'top_ngram_chr_fraction_2',
#  'top_ngram_chr_fraction_3',
#  'top_ngram_chr_fraction_4',
#  'oov_ratio',
 'dependency_distance_mean',
 'dependency_distance_std',
 'prop_adjacent_dependency_relation_mean',
 'prop_adjacent_dependency_relation_std']
# df_text[['y']+metrics.columns].corr(method='spearman')




import spacy
import textdescriptives as td
# load your favourite spacy model (remember to install it first using e.g. `python -m spacy download en_core_web_sm`)
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textdescriptives/all")  #TODO: dont compute coherence, quality, etc.


for split in ['train', 'test']:
	print('extracting', split)
	df_text = dfs[split]['df_text'].copy()
	# docs = df_text['text'].values
	
	doc = nlp.pipe(df_text['text'])
	td_features = td.extract_df(doc, include_text=False, metrics =["descriptive_stats","readability", 'quality', 'pos_proportions', 'dependency_distance'])
	

	td_features = td_features[td_columns] # only keep td_columns

	assert td_features.shape[0] == df_text.shape[0]
	dfs[split]['text_descriptives'] = td_features.copy()

	df_text_td = df_text.join(td_features, how="left")

	dfs[split]['srl_unvalidated_text_descriptives'] = df_text_td.drop(['conversation_id', 'y', 'text', 'word_count', 'Direct self-injury 2', 'Relationship issues 2'], axis=1).copy()



# In[316]:


import seaborn as sns
td_features = dfs['train']['text_descriptives']
td_features = td_features[td_columns]
td_features['y'] = dfs['train']['y'].values
td_features_corr = td_features.corr(method='spearman')
td_features_corr = td_features_corr.fillna(td_features_corr.median().median())

sns.set(font_scale=0.75)
sns.clustermap(td_features_corr)
# n = 5

# Adjust both x-tick and y-tick label sizes
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=n)  # Set the fontsize for x-tick labels
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=n)  # Set the fontsize for y-tick labels

# plt.show()


# # Extract embeddings
# 
# - 4000 docs - 8m
# 
# - 1000 docs - 1.5 m
# 

# In[171]:


# !pip install tensorboard


# In[346]:


len(dfs[split]['X']) == dfs[split]['df_text'].shape[0]


# In[349]:


run_this = True
# 25m for train set.
if run_this:
	import tensorboard
	from sentence_transformers import SentenceTransformer, util 
	embeddings_name = 'all-MiniLM-L6-v2'
	# Encode the documents with their sentence embeddings 
	# a list of pre-trained sentence transformers
	# https://www.sbert.net/docs/pretrained_models.html
	# https://huggingface.co/models?library=sentence-transformers
	
	# all-MiniLM-L6-v2 is optimized for semantic similarity of paraphrases
	sentence_embedding_model = SentenceTransformer(embeddings_name)       # load embedding
	
	sentence_embedding_model._first_module().max_seq_length = 500
	# TODO: Change max_seq_length to 500
	# Note: sbert will only use fewer tokens as its meant for sentences, 
	print(sentence_embedding_model .max_seq_length)



	for split in ['train', 'test']:
		dfs[split]['embeddings'] = sentence_embedding_model.encode(dfs[split]['X'], convert_to_tensor=True,show_progress_bar=True)
	
	# TODO move up to where I encoded this
		
	for split in ['train', 'test']:
		embeddings = dfs[split]['embeddings']
		embeddings = pd.DataFrame(embeddings, columns = [f'{embeddings_name}_{str(n).zfill(4)}' for n in range(embeddings.shape[1])])
		dfs[split][embeddings_name] = embeddings

	


# In[371]:





# # Load everything above

# In[372]:


import pickle
run_this = True #True saves, False loads
if run_this:
    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'wb') as f:
        pickle.dump(dfs, f) 
else:

    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'rb') as f:
    	dfs = pickle.load(f)

