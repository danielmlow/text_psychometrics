
# !module load openmind8/cuda/11.7
# !python --version #3.10.12

# !pip install --upgrade accelerate
# !pip install --upgrade transformers



import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import accelerate
import torch
from importlib import reload
print('accelerate', accelerate.__version__)
import transformers
print('transformers', transformers.__version__)

# Local
import api_keys
from huggingface_hub import login
login(token=api_keys.huggingface)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device    

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






import re


def create_binary_dataset(df_metadata, dv = 'suicide', n_per_dv = 3000):
    df_metadata_tag_1 = df_metadata[df_metadata[dv]==1].sample(n=n_per_dv,random_state=123)
    df_metadata_tag_0 = df_metadata[df_metadata[dv]==0].sample(n=n_per_dv,random_state=123)
    assert df_metadata_tag_1.shape[0] == n_per_dv
    assert df_metadata_tag_0.shape[0] == n_per_dv

    df_metadata_tag = pd.concat([df_metadata_tag_1, df_metadata_tag_0]).reset_index(drop=True)

    return df_metadata_tag


def obtain_json(responses):
    jsons = []
    for response in responses:
        matches = re.findall(r'\{.*?\}', response)

        # Assuming there's at least one match and it's safe to evaluate
        if matches:
            # Convert the first match to dictionary
            dictionary = eval(matches[0])
            jsons.append(dictionary)
        else:
            jsons.append(response)
    return jsons




pd.set_option("display.max_columns", None)

location = 'local'

if location == 'openmind':
  input_dir = '/nese/mit/group/sig/projects/dlow/ctl/datasets/'
  output_dir = 'home/dlow/'
elif location =='local':
  input_dir = '/Users/danielmlow/data/ctl/input/datasets/'
  output_dir = '/home/dlow/datum/lexicon/data/output/'


set_name = 'train10_test'	
test = pd.read_parquet(input_dir + f'{set_name}_metadata_messages.gzip', engine='pyarrow')


ctl_tags13 = ['self_harm',
 'suicide',
 'bully',
 'abuse_physical',
 'abuse_sexual',
 'relationship',
 'bereavement',
 'isolated',
 'anxiety', #anxiety_stress
 'depressed',
 'gender', #gender_sexual_identity
 'eating', # eating_body_image
 'substance']

# Clean
# ==============================
duplicates = test[test.duplicated(subset='message', keep=False)]
duplicates = duplicates.sort_values(by='message')
# Display duplicated rows
remove_duplicates = duplicates['message'].unique() # these are all short messages like test STOP, 
print(test.shape)
# remove rows if message is in remove_duplicates
test = test[~test ['message'].isin(remove_duplicates)]

# Remove if prank_ban? Yes.
[print('\n', n) for n in test[test['prank_ban']==1]['message_with_interaction'].tolist()]
test[test['prank_ban']==1][ctl_tags13].sum()
test[test['prank_ban']==1].shape
[print('\n', n) for n in test[test['prank']==1]['message_with_interaction'].tolist()]
test[test['prank']==1][ctl_tags13].sum()
test[test['prank']==1].shape
test = test[test['prank']!=1]

# Testing: yes, remove
[print('\n', n) for n in test[test['testing']==1]['message_with_interaction'].tolist()]
print(test.shape)
test[test['testing']==1].shape
test = test[test['testing']!=1]

print(test.shape)

# About someone else
[print('\n', n) for n in test[test['3rd_party']==1]['message_with_interaction'].tolist()]
test[test['3rd_party']==1][ctl_tags13].sum()
test[test['3rd_party']==1].shape

# TODO: redo ladder: -1, -2, -3 from third party, prank, test
def get_true_risk_3(row):
	if (row['3rd_party'] ==1 or row['testing'] == 1 or row['prank'] == 1):
		return -1
	elif (row['active_rescue'] > 0 or row['ir_flag'] > 0 or row['timeframe'] > 0):
		return 3 # high risk
	elif ('suicidal_desire' in row['suicidality'] or 'suicidal_intent' in row['suicidality'] or 'suicidal_capability' in row['suicidality']  or row['suicide']>0 or row['self_harm']>0): 
		return 2 # medium risk
	else:
		return 1 # normal risk
	


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
		


# What about none? It is mainly that they don't respond. So I should rmeove?
test[test ['none']==1][ctl_tags13+['none']].sum()
[print('\n', n) for n in test[test ['none']==1]['message_with_interaction'].tolist()[::100]]
test = test[test ['none']!=1] # Remove Nones


# Remove initial keyword
from collections import Counter
counter = Counter([str(n).split('\n')[0].lower() for n in test['message'].values])
ranked_results = counter.most_common()
remove_list = [n[0] for n in  ranked_results[:200]]
test['message'].duplicated().sum()
keep = ['hello', 'hi', 'hey', 'hello', 'i need help', 'help', 'hello?', 'test', 'i need help', 'hello.', 'hi.', 'help me', 'i need to talk to someone ', 'hola', 'i want to die', 'please help', 'hi i need help', 'i need to talk ', 'i need someone to talk to ' , 'i need help.', 'i need to talk to someone', 'i need to talk', 'is anyone there?', "hi, i'm going through something difficult and want to talk with someone who might be able to help."]
remove_list = [n for n in remove_list if n not in keep]
# remove first message unless in keep
# Process each message
processed_messages = []
for message in test['message'].values:
    message_split = message.split('\n')
    if message_split[0].lower() in remove_list:
        message_split.pop(0)  # Remove the first element
    if message_split[-1].lower() in ['stop']:
        # remove last element of list
        message_split.pop(-1)
    processed_messages.append('\n'.join(message_split))
test['message_clean'] = processed_messages



# TODO: Remove stop from end of 
processed_messages = []
for message in test['message_clean'].values:
    message_split = message.split('\n')
    if message_split[0].lower() in remove_list:
        message_split.pop(0)  # Remove the first element
    processed_messages.append('\n'.join(message_split))
test['message_clean'] = processed_messages
'home' in remove_list


# Remove short messages

messages = ['. '.join(n.split('\n')) for n in test['message_clean'].values]
word_count = [len(n.split(' ')) for n in messages]
import matplotlib.pyplot as plt

pd.DataFrame([len(n.split(' ')) for n in messages]).value_counts(normalize=True)
# what percentile of word_count is N?
n=10
import numpy as np
test['word_count'] = word_count
start = 10
end = 20
num = 123
[print('\n', n) for n in test[(test['word_count'] >start) & (test['word_count'] <end)]['message_with_interaction'].sample(n=20, random_state=num).values]
[print('\n', n) for n in test[(test['word_count'] >start) & (test['word_count'] <end)]['message_clean'].sample(n=20, random_state=num).values]
from scipy.stats import percentileofscore
percentileofscore(word_count,20, kind='mean')
plt.hist(word_count, bins=100)








for dv in ctl_tags13:
	df_metadata_tag = create_binary_dataset(test, dv = dv, n_per_dv = 250)
	display(df_metadata_tag[dv].value_counts())
	display(df_metadata_tag[ctl_tags13].sum())
	6 * 500 / 60 # minutes it will take to do inference with gemma 7b 




# TODO: See how they use it for text classification: (from probs or output layer directly?)
# https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb
from transformers import AutoTokenizer, AutoModelForCausalLM
max_length = 2000
model_name = "google/gemma-2b-it"
# model_name = "google/gemma-7b-it"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if 'gemma' in model_name:
	# Gemma
	model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
	model = model.to(device)
elif 'llama' in model_name:
	# Have to restart session after updating transformers
	from transformers import AutoTokenizer, LlamaForCausalLM
	model = LlamaForCausalLM.from_pretrained(model_name)

# !python3 -m nvitop -1
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




# documents = ['No one cares about me']
import time
documents = ['No one cares about me', "Help me please. Everything hurts. Why is it worse at night? God I hate everything. I can't really take it anymore. I'm so over it. I hate feeling like this, I don't WANT to feel like this. Thank you. My feelings.. Depression. I don't have clinical but lately I've been feeling like shit, especially at night. It really really is. I just want to be that happy optimistic person full of [scrubbed] that spreads confetti all over my friends and families lives. And I usually can be but I want to be like that all the time and I'm just so fucking tired of everything.. Yeah, that makes sense. It sounds about right. Yeah. Loneliness is something I feel a lot. I have amazing friends to vent to, and that I love and they love me, but I'm still really lonely. I guess confusion is a big feeling I feel as well. I'm lonely and confused. Yes. Drowning myself. Or stabbing myself. Yes I do. The Saturday coming up. No one will be home. Then I can call the cops and tell them I'm going to kill my self so my family doesn't have to see my body. Yes i am. About suicide? No. Your welcome.. Yeah. I don't need them to worry about me. Yeah I do. My family and friends are my world. My dad would. And my best friends [scrubbed] and [scrubbed]. That's something I wonder. Am I really their world. I'm kinda a bitch. Or I can be. Like I said, I try to be really upbeat, but I have some unflattering moments. If I'm their world? No. It seems a bit possessive. Well they're always here for me and willing to talk and let me vent, so that means a lot to me. Especially since we all know we can come to each other. I vent to them, they vent to me. Yeah, I think they do too. I'm not sure. Yes I think so. Maybe distractions. Like YouTube or something. Yeah I think so. I also write in my [scrubbed] a lot. That helps. The online chat thing sounds like it would be a good support system. Thank you so much. And yeah, I've reached out here multiple times. I'm feeling stable now.. That sounds good [scrubbed]. Thank you. Thank you so much."]
constructs = ['loneliness','suicide']
responses = []
time_elapsed_all = []
for document, construct in zip(documents, constructs):

	start = time.time()
	  




	prompt = """
	You are a text classification assistant.

	The text to classify:
	```
	{document}
	```

	Assign a probability for following labels and return in a JSON format:

	'related to {construct} at any point': <your_probability>, 'not related to {construct}': <your_probability>

	Do not provide additional text or explanations, just that JSON output.
	"""


	prompt = prompt.format(document = document, construct = construct)
	print(prompt)


	if 'gemma' in model_name:
		# Gemma
		input_ids = tokenizer(prompt, return_tensors="pt").to(device)
		outputs = model.generate(**input_ids, max_length = max_length)
		tokenizer.decode(outputs[0])
		# Find the length of the input_ids to know where the original prompt ends
		prompt_length = input_ids["input_ids"].shape[1]
		# Decode only the generated part, skipping the prompt
		response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
		responses.append(response)
	elif 'llama' in model_name:
		inputs = tokenizer(prompt, return_tensors="pt")

		# Generate
		generate_ids = model.generate(inputs.input_ids, max_length=max_length)
		response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		responses.append(response)
	
	end = time.time()
	time_elapsed = end - start
	time_elapsed_all.append(time_elapsed)

print(responses)










