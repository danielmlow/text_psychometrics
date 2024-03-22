#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install transformers==4.38.1
# !pip install accelerate==0.27.2


# In[2]:


# Config
exit_if_no_cuda = True #exit if cuda not found
# Set default values
toy = False
model_name = "google/gemma-2b-it"
with_interaction = True

# Check for arguments and assign them if they exist
import sys
if len(sys.argv) > 1:
	toy = sys.argv[1]
	if toy == 0:
		toy = False
	elif toy == 1:
		toy = True
if len(sys.argv) > 2:
	model_name = sys.argv[2]
if len(sys.argv) > 3:
	with_interaction = sys.argv[3]
	if with_interaction == 0:
		with_interaction = False
	elif with_interaction == 1:
		with_interaction = True




location = 'openmind'





# In[3]:


import os
import subprocess
import pandas as pd
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import random
from tqdm import tqdm
import random
import datetime
import numpy as np
from sklearn.metrics import classification_report
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from huggingface_hub import login

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	auc,
	confusion_matrix,
	f1_score,
	precision_recall_curve,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr


import api_keys # local

pd.set_option("display.max_columns", None)

if location == 'openmind':
  input_dir = '/nese/mit/group/sig/projects/dlow/ctl/datasets/'
  output_dir = './data/output/ml_performance/'
elif location =='local':
  input_dir = '/Users/danielmlow/data/ctl/input/datasets/'
  output_dir = '/home/dlow/datum/lexicon/data/output/ml_performance/'


set_name = 'train10_test'	
filename = f'{set_name}_metadata_messages_clean.gzip'
test = pd.read_parquet(input_dir + filename, engine='pyarrow')


# In[40]:


import logging
# Function to add a file handler to the root logger
def add_file_handler(log_filename):
	# Create a file handler that logs even debug messages
	fh = logging.FileHandler(log_filename, mode='a')
	fh.setLevel(logging.INFO)
	# Create formatter and add it to the handler
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	fh.setFormatter(formatter)
	# Remove all handlers associated with the root logger object.
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	# Add the handler to the root logger
	logging.root.addHandler(fh)

# Define your custom print function
def custom_print(*args, **kwargs):
	# Convert all arguments into a string. You might want to customize the separator.
	message = ' '.join(str(arg) for arg in args)
	# Log the message using logging
	logging.info(message)


# In[6]:


model_name_clean = model_name.replace('/', '-')
ts = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
output_dir_i = output_dir+f'{model_name_clean}_{ts}/' 
os.makedirs(output_dir_i , exist_ok=True)
# this will get replaced at inference time creating one for each DV
logging.basicConfig(filename=output_dir_i+f'log_print_statements_gpu_info.txt', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

print = custom_print


# In[ ]:


print(f"Toy: {toy}, Model Name: {model_name}, With Interaction: {with_interaction}")
print('running:', input_dir+filename)
print('location:', location)
print('\n')


# In[7]:


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

print('/n')
print('prompt_names:', prompt_names)


# In[8]:


login(token=api_keys.huggingface)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device    

if device =='cpu' exit_if_no_cuda:
	sys.exit()

# Check if CUDA is available
if torch.cuda.is_available():
	# Print number of GPUs available
	print("Number of GPUs available:", torch.cuda.device_count())

	for i in range(torch.cuda.device_count()):
		print(f"GPU {i}:")
		print(f"\tName: {torch.cuda.get_device_name(i)}")
		print(f"\tCuda version: {print(torch.version.cuda)}")
		print(f"\tCompute Capability: {torch.cuda.get_device_capability(i)}")
		print(f"\tTotal Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9} GB")
		print(torch.cuda.get_device_properties(i))
		# Additional details can be accessed via `torch.cuda.get_device_properties(i)`

else:
	print("CUDA is not available. Please check your installation and if your hardware supports CUDA.")
	print('/n')








# In[9]:


def run_nvitop():

	# Command to run 'nvitop' in one-shot mode
	command = ["python3", "-m", "nvitop", "-1"]

	# Run the command and capture its output

	try: 
		result = subprocess.run(command, capture_output=True, text=True)
		print(result)
		print(result.stdout)
	except: pass
	return


def create_binary_dataset(df_metadata, dv = 'suicide', n_per_dv = 3000):
	df_metadata_tag_1 = df_metadata[df_metadata[dv]==1].sample(n=n_per_dv,random_state=123)
	df_metadata_tag_0 = df_metadata[df_metadata[dv]==0].sample(n=n_per_dv,random_state=123)
	assert df_metadata_tag_1.shape[0] == n_per_dv
	assert df_metadata_tag_0.shape[0] == n_per_dv

	df_metadata_tag = pd.concat([df_metadata_tag_1, df_metadata_tag_0]).sample(frac=1).reset_index(drop=True)

	return df_metadata_tag




def find_json_in_string(string: str) -> str:
	"""Finds the JSON object in a string.

	Parameters
	----------
	string : str
		The string to search for a JSON object.

	Returns
	-------
	json_string : str
	"""
	start = string.find("{")
	end = string.rfind("}")
	if start != -1 and end != -1:
		json_string = string[start : end + 1]
	else:
		json_string = "{}"
	return json_string


# In[10]:


def cm(y_true, y_pred, output_dir, model_name, ts, classes = ["SITB-", "SITB+"], save=True):
	cm = confusion_matrix(y_true, y_pred, normalize=None)
	cm_df = pd.DataFrame(cm, index=classes , columns=classes )
	cm_df_meaning = pd.DataFrame([["TN", "FP"], ["FN", "TP"]], index=classes , columns=classes )

	cm_norm = confusion_matrix(y_true, y_pred, normalize="all")
	cm_norm = (cm_norm * 100).round(2)
	cm_df_norm = pd.DataFrame(cm_norm, index=classes , columns=classes )

	
	# plt.rcParams["figure.figsize"] = [4, 4]
	# ConfusionMatrixDisplay(cm_norm, display_labels=classes ).plot()
	# plt.tight_layout()
	
	if save:
		# plt.savefig(output_dir + f"cm_{model_name}_{ts}.png", dpi = 300)
		cm_df_meaning.to_csv(output_dir + f"cm_meaning_{model_name}_{ts}.csv")
		cm_df.to_csv(output_dir + f"cm_{model_name}_{ts}.csv")
		cm_df_norm.to_csv(output_dir + f"cm_norm_{model_name}_{ts}.csv")

	return cm_df_meaning, cm_df, cm_df_norm


def custom_classification_report(y_true, y_pred, y_pred_proba_1, output_dir,gridsearch=None,
										best_params=None,feature_vector=None,model_name=None,round_to = 2, ts = None, save_results=False, dv = None):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	np.set_printoptions(suppress=True)
	roc_auc = roc_auc_score(y_true, y_pred_proba_1)
	f1 = f1_score(y_true, y_pred)

	# calculate precision and recall for each threshold
	lr_precision, lr_recall, thresholds = precision_recall_curve(y_true, y_pred_proba_1)

	# TODO: add best threshold
	fscore = (2 * lr_precision * lr_recall) / (lr_precision + lr_recall)
	fscore[np.isnan(fscore)] = 0
	ix = np.argmax(fscore)
	best_threshold = thresholds[ix].item()

	pr_auc = auc(lr_recall, lr_precision)
	# AU P-R curve is also approximated by avg. precision
	# avg_pr = metrics.average_precision_score(y_true,y_pred_proba_1)

	sensitivity = recall_score(y_true, y_pred)
	specificity = tn / (tn + fp)  # OR: recall_score(y_true,y_pred, pos_label=0)
	precision = precision_score(y_true, y_pred)

	results = pd.DataFrame(
		[dv, feature_vector,model_name, sensitivity, specificity, precision, f1, roc_auc, pr_auc, best_threshold, gridsearch, best_params, len(y_true)],
		index=["Construct","Feature vector","Model", "Sensitivity", "Specificity", "Precision", "F1", "ROC AUC", "PR AUC", "Best th PR AUC", "Gridsearch", "Best parameters", "Support"],
	).T.round(2)
	if save_results:
		results.to_csv(output_dir + f"results_{model_name}_{ts}.csv")
	return results



def obtain_json(responses):

	jsons = []

	added = []

	for i, response in enumerate(responses):
		try:
			response_eval = eval(response)
			if type(response_eval) == dict:
				jsons.append(response_eval)
				added.append(i)
			elif type(response_eval) == set:
				jsons.append(list(response_eval))
				added.append(i)

		except:
			matches = re.findall(r'\{.*?\}', response)

			# Assuming there's at least one match and it's safe to evaluate
			if matches != []:
				# Convert the first match to dictionary
				try: 
					dictionary = eval(matches[0])
					jsons.append(dictionary)
					added.append(i)
				except:
					jsons.append(response)
					added.append(i)

			else:
				jsons.append(response)
				added.append(i)
	
		if i not in added:
			jsons.append(response)
		
	not_added = list(set(range(len(responses)))- set(added))
	if len(not_added)>1:
		print('WARNING: indexes not added, fix:', not_added)
	print('/n')
	return jsons


# # Load model (download if not in cache)

# In[11]:


# Erase model from session
# try: del tokenizer
# except: pass
# torch.cuda.empty_cache()
print('/n')
run_nvitop()
print('loading model...')



# In[12]:


# TODO: See how they use it for text classification: (from probs or output layer directly?)
# https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb


tokenizer = AutoTokenizer.from_pretrained(model_name)



if 'gemma' in model_name:
	# Gemma
	model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)

elif 'llama' in model_name:
	# Have to restart session after updating transformers
	from transformers import AutoTokenizer, LlamaForCausalLM
	model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
elif "paulml/OGNO-7B" in model_name:
	pipeline = transformers.pipeline(
	"text-generation",
	model=model_name,
	torch_dtype=torch.float16,
	device_map="auto",
	)
elif "microsoft/phi-2" in model_name:
	model = AutoModelForCausalLM.from_pretrained(model_name,  trust_remote_code=True,torch_dtype='auto', low_cpu_mem_usage=True)
	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


print('/n')
print('model loaded')    
run_nvitop()
# !nvidia-sim

	
	


# # alternative
# # Llama
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# sequences = pipeline(
#     prompt,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")


# In[13]:


# get a sense of token length vs word length
# ========================================================
run_this =False

if run_this:

	# for dv in ctl_tags13:
	dv = 'eating'
	df_i = create_binary_dataset(test, dv = dv, n_per_dv = 300)
	df_i[ctl_tags13].sum()


	documents = df_i['message_with_interaction_clean'].values
	y_test = df_i[dv].values
	construct = prompt_names.get(dv)
	print(construct)
	print(len(documents))
	print(len(y_test))

	len_tokens = []
	word_counts = []

	for text in documents:
		# Tokenize text
		tokens = tokenizer.tokenize(text)

		# Number of tokens
		len_tokens.append(len(tokens))
		word_counts.append(len(text.split(' ')))
		
	plt.hist(len_tokens, bins = 100, alpha =0.3)
	plt.hist(word_counts, bins = 100, alpha =0.3, label = 'word counts')
	plt.legend()
	print(np.mean(len_tokens), np.mean(word_counts))


# In[14]:


# batch tokenization was messing up my memory usage even though it doesnt use much memory.
# ================================================================================================
# max_length = 1012
# prompts = [prompt.format(context = 'Crisis Text Line service', document = document, construct = construct) for document, construct in zip(documents, constructs)] 
# input_ids_all = tokenizer(prompts, padding=True, truncation=True,max_length = max_length, return_tensors="pt").input_ids.to(device)

# start = time.time()

# outputs = model.generate(input_ids_all, max_length=max_length)
# # Decode generated sequences
# prompt_length = input_ids_all.shape[1]
# # # Decode only the generated part, skipping the prompt
# generated_texts = [tokenizer.decode(output_sequence[prompt_length:], skip_special_tokens=True) for output_sequence in outputs]
# end = time.time()
# time_elapsed = end - start
# print(time_elapsed)


# In[15]:


model_name


# In[57]:

construct_position = 0

prompts ={ 
	'google/gemma-2b-it': 
"""
You are a conversation classification assistant. Classify the following {context} conversation:

Here is the {context} conversation (ends with ```):
```
{document}
```

Assign probabilities for the following labels and return using this JSON format (do not provide additional notes, explanations, or warnings). Provide your best guess, only return JSON (both probabilities should sum to 1):

JSON:
{{'texter mentions something related to {construct}': <your_probability>, 'texter does not mention anything related to {construct}': <your_probability>}}


JSON:
""",
	
	
	'google/gemma-7b-it': 
"""
You will classify a {context} conversation as to whether it contains any content related to {construct}.

Here is the conversation (begins and ends with ```):
```
{document}
```

Assign probabilities only for following two labels and return using this JSON format (do not provide additional notes, explanations, or warnings). Only return JSON:

JSON:
{{'contains any content related to {construct}': <your_probability>, 'does not contain any content related to {construct}': <your_probability>}}

JSON:
"""
	
	
#         'google/gemma-7b-it': 
# """
# You will classify a {context} conversation as to whether the texter is concerned about something related to {construct}.

# Here is the conversation (begins and ends with ```):
# ```
# {document}
# ```

# Assign probabilities only for following two labels and return using this JSON format (do not provide additional notes, explanations, or warnings). Only return JSON:

# JSON:
# {{'texter is concerned about something related to {construct}': <your_probability>, 'texter is not concerned about something related to {construct}': <your_probability>}}

# JSON:
# """
	
	
	
}

prompt = prompts.get(model_name)
print('/n')
print('prompt:\n', prompt, '\n')
print('/n')


# In[59]:


ctl_tags13


# In[ ]:


# if toy:
# 	n_per_dv = 75
# else:
n_per_dv = 300


if with_interaction:
	max_length = int(1750*1.4)+75 #word count * 1.4 +75 for the prompt ~ tokens, 98%have less than this
else:
	# just texter     
	max_length = int(1000*1.4)+75
	
# documents = ['No one cares about me']


# Accessing tokenized ids
for dv in ctl_tags13:
	output_dir_i_dv = output_dir_i+f'{dv}/'
	os.makedirs(output_dir_i_dv, exist_ok = True)
		

	responses = []
	time_elapsed_all = []
	add_file_handler(output_dir_i_dv+f'log_print_statements_{dv}.txt')
#     print = custom_print
	
	construct = prompt_names.get(dv)
	# Configure logging

	df_i = create_binary_dataset(test, dv = dv, n_per_dv = n_per_dv)
	
	if with_interaction:
		documents = df_i['message_with_interaction_clean'].values
	else:
		documents = df_i['message_clean'].values
	y_test = df_i[dv].values
	print('\n', dv, '============================================')
	print(df_i[ctl_tags13].sum())
	print('construct:', construct)
	print('len of documents:',len(documents))
	print('len of y_test:',len(y_test))

	for document, y_test_i in tqdm(zip(documents, y_test)):

		start = time.time()

		prompt_i = prompt.format(context = 'Crisis Text Line service', document = document, construct = construct)
		# print(prompt_i)


		if 'gemma' in model_name:
			# Gemma
			input_ids = tokenizer(prompt_i,truncation=True,max_length=max_length, return_tensors="pt").to(device)
			outputs = model.generate(**input_ids, max_new_tokens = 1000)
			tokenizer.decode(outputs[0])
			# Find the length of the input_ids to know where the original prompt ends
			prompt_length = input_ids["input_ids"].shape[1]
			# Decode only the generated part, skipping the prompt
			response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

		elif 'llama' in model_name:
			inputs = tokenizer(prompt_i, return_tensors="pt")

			# Generate
			generate_ids = model.generate(inputs.input_ids, max_length=max_length)
			response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

	#     elif "paulml/OGNO-7B" in model_name:
	#         messages = [{"role": "user", "content": prompt}]
	#         prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	#         outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
	#         response = outputs[0]["generated_text"] 
	#     elif 'microsoft/phi-2' in model_name:     
	#         inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

	#         outputs = model.generate(**inputs, max_length=200)
	#         response = tokenizer.batch_decode(outputs)[0]




		responses.append(response)

		print('y_test_i', y_test_i, '=======')
		print(response)


		end = time.time()
		time_elapsed = end - start
		# print(time_elapsed)
		time_elapsed_all.append(time_elapsed)

	# clean
	jsons = obtain_json(responses)


	# TODO: you need to randomly assign 0.2 or 0.8 if you can't parse a response
	could_not_parse = []
	json_responses_clean = []
	for i, response in enumerate(jsons):
		if type(response) == dict:
			response_values = response.values()
			json_responses_clean.append(list(response_values))
			could_not_parse.append(0)
		elif type(response) == list:
			json_responses_clean.append(response)
			could_not_parse.append(0)
			

		else:
			random_float_0 = random.uniform(0.51, 0.99)
			random_float_1 = 1 - random_float_0
			selected_list = random.choice([
				[random_float_0,random_float_1],
				[random_float_1,random_float_0],
				])
			print('\n\n=======', i, 'could not parse, randomly assiging a value:', response)
			json_responses_clean.append(selected_list)
			could_not_parse.append(1)


	if construct_position == 0: 
		labels_order = [f"{dv}", f"Other"]
	else:
		labels_order = [f"Other",f"{dv}"]
	y_pred_df = pd.DataFrame(json_responses_clean, columns = labels_order)
	y_pred_df['could_not_parse'] = could_not_parse
	y_pred_df['jsons'] = jsons
	y_pred_df['time_elapsed'] = time_elapsed_all

	y_pred_proba_1 = [n[construct_position] for n in json_responses_clean] # 1 value is construct
	y_pred_proba_1
	y_pred = np.array([n>=0.5 for n in y_pred_proba_1])*1  # 1 if construct >=0.5 independent of order in json_responses_clean

	y_pred_df['y_pred'] = y_pred
	y_pred_df['y_test'] = y_test
	
	y_pred_df.to_csv(output_dir_i_dv+f'y_proba_{dv}.csv')

	# here we don't change label orders because y_pred and y_test are well defined (1 if construct >=0.5)     
	dv_clean = dv.replace('_', ' ').capitalize()
	cm_df_meaning, cm_df, cm_df_norm = cm(y_test, y_pred, output_dir_i_dv, model_name_clean, ts, classes = [f"Other",f"{dv_clean}"], save=True)
	cr = classification_report(y_test, y_pred,output_dict=True)
	cr = pd.DataFrame(cr)

	cr.to_csv(output_dir_i_dv+f'cr_{dv}_{ts}.csv')

	results = custom_classification_report(y_test, y_pred, y_pred_proba_1, output_dir_i_dv,gridsearch=None,
		best_params=None,feature_vector=None,model_name=model_name_clean,round_to = 2, ts =ts, save_results=True, dv = dv_clean)







# In[ ]:


cr


# In[ ]:


results


# In[ ]:


cm


# In[ ]:




