#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!pip install -q pyarrow


# # Load dataset

# In[268]:


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

import pickle
run_this = False #True saves, False loads
if run_this:
    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'wb') as f:
        pickle.dump(dfs, f) 
else:

    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'rb') as f:
    	dfs = pickle.load(f)


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


# # Models

# In[373]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os 
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler

from lightgbm import LGBMClassifier # TODO: add
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
import warnings
from sklearn.preprocessing import StandardScaler
# !pip install xgboost
# !pip install lightgbm==4.3.0
from lightgbm import LGBMRegressor
import string
from sklearn.linear_model import Lasso
# import contractions # TODO: add
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import sys
sys.path.insert(1,'./../../concept-tracker')
from concept_tracker.utils import metrics_report # local script

from scipy.stats import pearsonr, spearmanr
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV # had to replace np.int for in in transformers.py
from importlib import reload
reload(metrics_report)
from concept_tracker.utils.metrics_report import cm, custom_classification_report, regression_report, generate_feature_importance_df
from sklearn import metrics
# from imblearn.pipeline import Pipeline as imb_Pipeline

# from imblearn.over_sampling import RandomOverSampler
import datetime

import nltk
nltk.download('stopwords')


ridge_alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_alphas_toy = [0.1, 10]
def get_params(feature_vector,model_name = 'Ridge', toy=False):
	if model_name in ['LogisticRegression']:
		if feature_vector == 'tfidf':
			if toy:
				warnings.warn('WARNING, running toy version')
				param_grid = {
				   'vectorizer__max_features': [256, 512],
				}
			else:
				param_grid = {
					'vectorizer__max_features': [512,2048,None],
					'model__C': ridge_alphas,
				}
	
		else:
			if toy:
				warnings.warn('WARNING, running toy version')
				param_grid = {
					'model__C': ridge_alphas_toy,
				}
			else:
				param_grid = {
					'model__C': ridge_alphas,
				}
	
	elif model_name in ['Ridge', 'Lasso']:
		if feature_vector == 'tfidf':
			if toy:
				warnings.warn('WARNING, running toy version')
				param_grid = {
				   'vectorizer__max_features': [256, 512],
				}
			else:
				param_grid = {
					'vectorizer__max_features': [512,2048,None],
					'model__alpha': ridge_alphas,
				}
	
		else:
			if toy:
				warnings.warn('WARNING, running toy version')
				param_grid = {
					'model__alpha': ridge_alphas_toy,
				}
			else:
				param_grid = {
					'model__alpha': ridge_alphas,
				}
	

	elif model_name in [ 'LGBMRegressor', 'LGBMClassifier']:
		if toy:
			warnings.warn('WARNING, running toy version')
			param_grid = {
			   # 'vectorizer__max_features': [256,2048],
				# 'model__colsample_bytree': [0.5, 1],
				'model__max_depth': [10,20], #-1 is the default and means No max depth
		
			}
		else:
			if feature_vector =='tfidf':
				param_grid = {
					'vectorizer__max_features': [256,2048,None],
					'model__num_leaves': [30,45,60],
					'model__colsample_bytree': [0.1, 0.5, 1],
					'model__max_depth': [0,5,15], #0 is the default and means No max depth
					'model__min_child_weight': [0.01, 0.001, 0.0001],
					'model__min_child_samples': [10, 20,40], #alias: min_data_in_leaf
				   'vectorizer__max_features': [256, 512],
					}
			
			param_grid = {
				'model__num_leaves': [30,45,60],
				'model__colsample_bytree': [0.1, 0.5, 1],
				'model__max_depth': [0,5,15], #0 is the default and means No max depth
				'model__min_child_weight': [0.01, 0.001, 0.0001],
				'model__min_child_samples': [10, 20,40], #alias: min_data_in_leaf
		
			}

	
	elif model_name in [ 'XGBRegressor', 'XGBClassifier']:
		if toy:
			warnings.warn('WARNING, running toy version')
			param_grid = {
				'model__max_depth': [10,20], #-1 is the default and means No max depth
		
			}
		else:
			if feature_vector =='tfidf':
				param_grid = {
					'vectorizer__max_features': [256,2048,None],
					'model__colsample_bytree': [0.1, 0.5, 1],
					'model__max_depth': [5,15, None], #None is the default and means No max depth
					'model__min_child_weight': [0.01, 0.001, 0.0001],
				
				   
					}
			
			param_grid = {
				'model__colsample_bytree': [0.1, 0.5, 1],
				'model__max_depth': [5,15, None], #None is the default and means No max depth
				'model__min_child_weight': [0.01, 0.001, 0.0001],
		
			}

	return param_grid

from sklearn.impute import SimpleImputer

def get_pipelines(feature_vector, model_name = 'Ridge'):
	
	# model = getattr(__main__, model_name)()
	model = globals()[model_name]()
	# if model_name == 'Ridge':
	#     model = Ridge()
	# elif model_name == 'XGBRegressor':
	#     model = XGBRegressor()
	model.set_params(random_state = 123)
	
	
	if feature_vector =='tfidf':
		pipeline = Pipeline([
			 ('vectorizer', vectorizer),
			 ('model', model), 
			])
	else:
		pipeline = Pipeline([
			('imputer', SimpleImputer(strategy='median')),
			('standardizer', StandardScaler()),
			 ('model', model), 
			])
	return pipeline


from sklearn import metrics



def tfidf_feature_importances(pipe, top_k = 100, savefig_path = '', model_name_in_pipeline = 'model', xgboost_method = 'weight' ):
    # # Using sklearn pipeline:
    feature_names = pipe.named_steps["vectorizer"].get_feature_names_out()
    
    try: coefs = pipe.named_steps["model"].coef_.flatten() # Get the coefficients of each feature
    except: 
        try: coefs = list(pipe.named_steps[model_name_in_pipeline].get_booster().get_score(importance_type=xgboost_method )) # pipeline directly
        except:
            # gridsearchcv(pipeline)
            coefs = pipe.best_estimator_.named_steps[model_name_in_pipeline].get_booster().get_score(importance_type=xgboost_method )
    
    # Without sklearn pipeline
    # feature_names = vectorizer.get_feature_names_out()
    # print(len(feature_names ))
    # coefs = pipeline.coef_.flatten() # Get the coefficients of each feature
    
    # Visualize feature importances
    # Sort features by absolute value
    df = pd.DataFrame(zip(feature_names, coefs), columns=["feature", "value"])
    df["abs_value"] = df["value"].apply(lambda x: abs(x))
    df["colors"] = df["value"].apply(lambda x: "orange" if x > 0 else "dodgerblue")
    df = df.sort_values("abs_value", ascending=False) # sort by absolute coefficient value
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 6))
    plt.style.use('default')  # Example of applying the 'ggplot' style
    ax = sns.barplot(x="value",
                y="feature",
                data=df.head(top_k),
                hue="colors")
    ax.legend_.remove()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_title(f"Top {top_k} Features", fontsize=14)
    ax.set_xlabel("Coef", fontsize=12) # coeficient from linear model
    ax.set_ylabel("Feature Name", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(savefig_path+'.png', dpi=300)
    plt.show()
    return df


# In[379]:


metrics_report = 1
from concept_tracker.utils import metrics_report


# tfidf 

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def nltk_lemmatize(text):
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]

# Now, integrate this with TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk_lemmatize, stop_words='english')

from sklearn.linear_model import Ridge

def custom_tokenizer(string):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(string)
    return words

def tokenizer_remove_punctuation(text):
    return re.split("\\s+",text)

vectorizer = TfidfVectorizer(
                 min_df=3, ngram_range=(1,2), 
                 stop_words=None, #'english',# these include 'just': stopwords.words('english')+["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'need', 'sha', 'wa', 'wo', 'would'], strip_accents='unicode',
                 sublinear_tf=True,
                 # tokenizer=nltk_lemmatize,
                token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'",
                    use_idf=True,
                 )






# def get_splits(feature_vector):
# 	if feature_vector in ['tfidf']:
# 		X_train = dfs['train']['X'] # text
# 		# X_val = dfs['val']['X']
# 		X_test = dfs['test']['X']
# 		y_train = dfs['train']['y']
# 		# y_val = dfs['val']['y']
# 		y_test = dfs['test']['y']
		
# 	elif feature_vector in ['liwc22']:        
		
# 		X_train = dfs['train']['liwc22_X'] 
# 		# X_val = dfs['val']['liwc22_X']    
# 		X_test = dfs['test']['liwc22_X']
# 		y_train = dfs['train']['liwc22_y']
# 		# y_val = dfs['val']['liwc22_y']
# 		y_test = dfs['test']['liwc22_y']

# 	elif feature_vector in ['srl_unvalidated']:        
		
# 		X_train = dfs['train']['srl_unvalidated'] 
# 		# X_val = dfs['val']['srl_unvalidated']    
# 		X_test = dfs['test']['srl_unvalidated']
# 		y_train = dfs['train']['y']
# 		# y_val = dfs['val']['y'] 
# 		y_test = dfs['test']['y']

# 	elif feature_vector in ['SRL GPT-4 Turbo']:
# 		X_train = dfs['train']['SRL GPT-4 Turbo'][constructs_in_order] 
# 		# X_val = dfs['val']['SRL GPT-4 Turbo'][constructs_in_order]    
# 		X_test = dfs['test']['SRL GPT-4 Turbo'][constructs_in_order]
# 		y_train = dfs['train']['y']
# 		# y_val = dfs['val']['y'] 
# 		y_test = dfs['test']['y']
		

# 	elif feature_vector in ['text_descriptives']:        
		
# 		X_train = dfs['train']['text_descriptives'] 
# 		X_test = dfs['test']['text_descriptives']
# 		y_train = dfs['train']['y']
# 		y_test = dfs['test']['y']
		
# 	elif feature_vector in ['srl_unvalidated_text_descriptives']:        
		
# 		X_train = dfs['train']['srl_unvalidated_text_descriptives'] 
# 		X_test = dfs['test']['srl_unvalidated_text_descriptives']
# 		y_train = dfs['train']['y']
# 		y_test = dfs['test']['y']
	

	
# 	elif feature_vector in ['all-MiniLM-L6-v2']:
# 		X_train = dfs['train']['all-MiniLM-L6-v2'] 
# 		# X_val = dfs['val']['all-MiniLM-L6-v2']    
# 		X_test = dfs['test']['all-MiniLM-L6-v2']
# 		y_train = dfs['train']['y']
# 		# y_val = dfs['val']['y']
# 		y_test = dfs['test']['y']
		
	
# 	return X_train, y_train,X_test, y_test


def custom_classification_report(y_true, y_pred, y_pred_proba_1, output_dir,gridsearch=None,
										best_params=None,feature_vector=None,model_name=None,round_to = 2, ts = None, save_csv=False ):
	
	if len(np.unique(y_true)) == 1:
		sensitivity = metrics.recall_score(y_true, y_pred)
		# Calculate TP and FN
		TP = sum((y_pred == 1) & (y_true == 1))
		FN = sum((y_pred == 0) & (y_true == 1))

		# Now you can calculate the False Negative Rate (FNR)
		fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

		results = pd.DataFrame(
			[feature_vector,model_name, sensitivity, np.nan, np.nan,fnr,  np.nan, np.nan, np.nan, np.nan, gridsearch, best_params],
			index=["Feature vector","Model", "Sensitivity", "Specificity", "Precision", "FNR", "F1", "ROC AUC", "PR AUC", "Best th PR AUC", "Gridsearch", "Best parameters"],
		).T.round(2)			


	
	else:
		tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
		fnr = fn / (fn + tp)
		np.set_printoptions(suppress=True)
		roc_auc = roc_auc_score(y_true, y_pred)
		f1 = f1_score(y_true, y_pred)

		# calculate precision and recall for each threshold
		lr_precision, lr_recall, thresholds = precision_recall_curve(y_true, y_pred_proba_1)

		# fscore = (2 * lr_precision * lr_recall) / (lr_precision + lr_recall)
		# fscore[np.isnan(fscore)] = 0
		# ix = np.argmax(fscore)
		# best_threshold = thresholds[ix].item()
		best_threshold = np.nan

		pr_auc = auc(lr_recall, lr_precision)
		# AU P-R curve is also approximated by avg. precision
		# avg_pr = metrics.average_precision_score(y_true,y_pred_proba_1)

		sensitivity = metrics.recall_score(y_true, y_pred)
		specificity = tn / (tn + fp) # metrics.recall_score(y_true,y_pred, pos_label=0)   # 
		precision = metrics.precision_score(y_true, y_pred)

		results = pd.DataFrame(
			[feature_vector,model_name, sensitivity, specificity, precision,fnr,  f1, roc_auc, pr_auc, best_threshold, gridsearch, best_params],
			index=["Feature vector","Model", "Sensitivity", "Specificity", "Precision","FNR", "F1", "ROC AUC", "PR AUC", "Best th PR AUC", "Gridsearch", "Best parameters"],
		).T.round(2)

	

	if save_csv:
		results.to_csv(output_dir + f"results_{model_name}_{ts}.csv")
	return results




# In[385]:


from itertools import product


parameters =   {'model__colsample_bytree': [1, 0.5, 0.1],
                'model__max_depth': [-1,10,20], #-1 is the default and means No max depth
                'model__min_child_weight': [0.01, 0.001, 0.0001],
                'model__min_child_samples': [10, 20,40], #alias: min_data_in_leaf
               }
        

combinations = list(product(*parameters.values()))
        
def get_combinations(parameters):
    
    parameter_set_combinations = []
    for combination in combinations:
        parameter_set_i = {}
        
        for i, k in enumerate(parameters.keys()):
            parameter_set_i[k] = combination[i]
        parameter_set_combinations.append(parameter_set_i)
    return parameter_set_combinations



run_this = False #True saves, False loads
if run_this:
    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'wb') as f:
        pickle.dump(dfs, f) 
else:

    with open(f'./data/input/ctl/ctl_dfs_features_{task}.pkl', 'rb') as f:
    	dfs = pickle.load(f)




# config
# ========================================================================================
toy = False



feature_vectors = ['liwc22', 'srl_validated'] #, 'liwc22_semantic']#, ]#['all-MiniLM-L6-v2', 'srl_unvalidated','SRL GPT-4 Turbo', 'liwc22', 'liwc22_semantic'] # srl_unvalidated_text_descriptives','text_descriptives' ]
sample_sizes = [50, 150, 2000] 

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




if toy:
	sample_sizes = [50]
	feature_vectors = feature_vectors[:2]

for n in sample_sizes:
	results = []
	results_content_validity = []
	# for gridsearch in [True]:

	# for feature_vector in ['srl_unvalidated', 'all-MiniLM-L6-v2']:#['srl_unvalidated']:#, 'srl_unvalidated']:
	for feature_vector in feature_vectors:#['srl_unvalidated']:#, 'srl_unvalidated']:
		X_test_3 = dfs['X_test_content_validity_prototypicality-3'][feature_vector]
		X_test_13 = dfs['X_test_content_validity_prototypicality-1_3'][feature_vector]
		

		if toy:
			output_dir_i = output_dir + f'results_{ts_i}_toy/'
		else:
			output_dir_i = output_dir + f'results_{ts_i}_{n}/'
			
		os.makedirs(output_dir_i, exist_ok=True)
		
		for dv in ctl_tags13:
			
			
			responses = []
			
			time_elapsed_all = []
		
			train_i = create_binary_dataset(train, dv = dv, n_per_dv = n)
			test_i = create_binary_dataset(test, dv = dv, n_per_dv = 300)

			
			y_test =  test_i[dv].values

			train_i_y = train_i[['conversation_id', dv]]

			test_i_y = test_i[['conversation_id', dv]]
			
			# print(len(train_i), len(test_i))
			# print(np.sum(y_train), np.sum(y_test))
			
			convo_ids_train = train_i['conversation_id'].values
			convo_ids_test = test_i['conversation_id'].values

			if 'srl' in feature_vector:
				feature_names = constructs_in_order+['word_count']
			elif feature_vector == 'liwc22':
				feature_names = liwc_nonsemantic+liwc_semantic
			elif feature_vector == 'liwc22_semantic':
				feature_names = liwc_semantic
		
			X_train_df_features = dfs['train'][feature_vector].copy()
			X_train_df_features =  X_train_df_features[X_train_df_features['conversation_id'].isin(convo_ids_train)]
			X_train_df_features = X_train_df_features.drop(dv,axis=1)
			train_i_y_X = pd.merge(train_i_y, X_train_df_features, on='conversation_id')
			y_train =  train_i_y_X[dv].values
			X_train = train_i_y_X[feature_names]
			
			print(y_train.shape, X_train.shape)


			X_test_df_features = dfs['test'][feature_vector].copy()
			X_test_df_features =  X_test_df_features[X_test_df_features['conversation_id'].isin(convo_ids_test)]
			X_test_df_features = X_test_df_features.drop(dv,axis=1)
			test_i_y_X = pd.merge(test_i_y, X_test_df_features, on='conversation_id')
			y_test =  test_i_y_X[dv].values
			X_test = test_i_y_X[feature_names]
			
			# content validity test sets
			X_test_13_dv = X_test_13[X_test_13['y_test']==dv][feature_names]
			y_test_13_dv = [1]*len(X_test_13_dv)

			X_test_3_dv = X_test_3[X_test_3['y_test']==dv][feature_names]
			y_test_3_dv = [1]*len(X_test_3_dv)



			# TODO: compute metrics for it considering only false positives. 
			
			# if feature_vector == 'liwc22_semantic':
			# 	X_train, y_train, X_test, y_test = get_splits('liwc22')
			# 	X_train = X_train[liwc_semantic]
			# 	# X_val = X_val[liwc_semantic]
			# 	X_test = X_test[liwc_semantic]
		
			# else:
			# 	X_train, y_train,X_test, y_test = get_splits(feature_vector)

			if toy:
				X_train['y'] = y_train
				X_train = X_train.sample(n = 100)
				y_train = X_train['y'].values
				X_train = X_train.drop('y', axis=1)

				
		
	
			
		

			# if task == 'classification':
			# 	encoder = LabelEncoder()

			# 	# Fit and transform the labels to integers
			# 	y_train = encoder.fit_transform(y_train)
			# 	y_test = encoder.transform(y_test)

			
			for model_name in model_names: 
		
				pipeline = get_pipelines(feature_vector, model_name = model_name)
				print(pipeline)
			
				# if gridsearch == 'minority':
				# 	# Obtain all hyperparameter combinations
				# 	parameters = get_params(feature_vector,model_name=model_name, toy=toy)
				# 	parameter_set_combinations = get_combinations(parameters)
				# 	scores = {}
				# 	for i, set in enumerate(parameter_set_combinations):
				# 		pipeline.set_params(**set)
				# 		pipeline.fit(X_train,y_train)
				# 		y_pred = pipeline.predict(X_val) # validation set 
				# 		rmse_per_value = []
				# 		rmse = metrics.mean_squared_error(y_val, y_pred, squared=False ) # validation set 
				# 		for value in np.unique(y_val):
				# 			y_pred_test_i = [[pred,test] for pred,test in zip(y_pred,y_val) if test == value] # validation set 
				# 			y_pred_i = [n[0] for n in y_pred_test_i]
				# 			y_test_i = [n[1] for n in y_pred_test_i]
				# 			rmse_i = metrics.mean_squared_error(y_test_i, y_pred_i, squared=False )
				# 			rmse_per_value.append(rmse_i )
				# 		scores[i] = [rmse]+rmse_per_value+[str(set)]
				# 	scores = pd.DataFrame(scores).T
				# 	scores.columns = ['RMSE', 'RMSE_2', 'RMSE_3', 'RMSE_4', 'Parameters']
				# 	scores = scores.sort_values('RMSE_4')
				# 	best_params = eval(scores['Parameters'].values[0])
				# 	pipeline.set_params(**best_params)
				# 	pipeline.fit(X_train,y_train)
				# 	y_pred = pipeline.predict(X_test)
					
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
				dv_clean = dv.replace(' ','_').capitalize()
				if task == 'classification':
					
					
					if gridsearch:
						y_proba = best_model.predict_proba(X_test)       # Get predicted probabilities
						y_pred_content_validity_13 = best_model.predict(X_test_13_dv)
						y_pred_content_validity_3 = best_model.predict(X_test_3_dv)
					else:

						y_proba = pipeline.predict_proba(X_test)       # Get predicted probabilities
						y_pred_content_validity_13 = pipeline.predict(X_test_13_dv)
						y_pred_content_validity_3 = pipeline.predict(X_test_3_dv)
					y_proba_1 = y_proba[:,1]
					y_pred = y_proba_1>=0.5*1                   # define your threshold
					# Predictions
					y_pred_df = pd.DataFrame(y_pred)
					
					y_pred_df.to_csv(output_dir_i+f'y_pred_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}_{dv}.csv', index=False)
					pd.DataFrame(y_pred_content_validity_13).to_csv(output_dir_i+f'y_pred_content_validity_13_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}_{dv}.csv', index=False)
					pd.DataFrame(y_pred_content_validity_3).to_csv(output_dir_i+f'y_pred_content_validity_3_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}_{dv}.csv', index=False)

					
					path = output_dir_i + f'{feature_vector}_{model_name}_{n}_{ts_i}_{dv}'
					cm_df_meaning, cm_df, cm_df_norm = cm(y_test,y_pred, output_dir_i, f'{model_name}_{dv}', ts_i, classes = ['Other', f'{dv_clean}'], save=True)
					results_i = custom_classification_report(y_test, y_pred, y_proba_1, output_dir_i,gridsearch=gridsearch,
											best_params=best_params,feature_vector=feature_vector,model_name=f'{model_name}_{dv}',round_to = 2, ts = ts_i)
					
					
					

					results_i_content_13 = custom_classification_report(y_test_13_dv, y_pred_content_validity_13, y_pred_content_validity_13, output_dir_i,gridsearch=gridsearch,
											best_params=best_params,feature_vector=feature_vector,model_name=f'{model_name}_{dv}_content-validity-13',round_to = 2, ts = ts_i)
					results_i_content_3 = custom_classification_report(y_test_3_dv, y_pred_content_validity_3, y_pred_content_validity_3, output_dir_i,gridsearch=gridsearch,
											best_params=best_params,feature_vector=feature_vector,model_name=f'{feature_vector}_{model_name}_{dv}_content-validity-3',round_to = 2, ts = ts_i)
				elif task == 'regression':
					if gridsearch:
						y_pred = best_model.predict(X_test)
					else:
						y_pred = pipeline.predict(X_test)

					results_i =regression_report(y_test,y_pred,y_train=y_train,
											metrics_to_report = metrics_to_report,
												gridsearch=gridsearch,
											best_params=best_params,feature_vector=feature_vector,model_name=model_name, plot = True, save_fig_path = path,n = n, round_to = 2)
				# results_i.to_csv(output_dir_i + f'results_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}_{dv}.csv')
				display(results_i)
				results.append(results_i)
				results_content_validity.append(results_i_content_13)
				results_content_validity.append(results_i_content_3)
				# Feature importance
				if feature_vector == 'tfidf':
					if model_name in ['XGBRegressor']:
						warnings.warn('Need to add code to parse XGBoost feature importance dict')
					else:
						feature_importances = tfidf_feature_importances(pipeline, top_k = 50, savefig_path = output_dir_i + f'feature_importance_{feature_vector}_{model_name}_{n}_{ts_i}_{dv}')
				else:
					feature_names = X_train.columns
					# TODO add correlation with DV to know direction
					feature_importance = generate_feature_importance_df(pipeline, model_name,feature_names,  xgboost_method='weight', model_name_in_pipeline = 'model')
					if str(feature_importance) != 'None':       # I only implemented a few methods for a few models
						feature_importance.to_csv(output_dir_i + f'feature_importance_{feature_vector}_{model_name}_gridsearch-{gridsearch}_{n}_{ts_i}_{dv}.csv', index = False)        
						# display(feature_importance.iloc[:50])
				
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
	


