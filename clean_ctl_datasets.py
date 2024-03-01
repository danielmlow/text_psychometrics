import pandas as pd

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
processed_messages_wi = []
last_messages = []
for message, message_with_interaction in zip(test['message'].values, test['message_with_interaction'].values):
	message_split = message.split('\n')
	message_split_wi = message_with_interaction.split('\n')
	if message_split[0].lower() in remove_list:
		message_split.pop(0)  # Remove the first element
		message_split_wi.pop(0)
	if len(message_split)>0:
		last_messages.append(message_split[-1].lower())
		if 'stop' in message_split[-1].lower():
			# remove last element of list
			message_split.pop(-1)
			message_split_wi.pop(-1)
	processed_messages.append('\n'.join(message_split))
	processed_messages_wi.append('\n'.join(message_split_wi))
test['message_clean'] = processed_messages
test['message_with_interaction_clean'] = processed_messages_wi

Counter(last_messages).most_common(50) # I removed stop, other ones can stay since they seem like normal endings

# Remove espanol
len([n for n in test['message_clean'].values if 'español'	in n.lower()])
print(test.shape)
test = test[~test['message_clean'].str.contains('español', case= False)]
print(test.shape)

# Remove short messages
messages = ['. '.join(n.split('\n')) for n in test['message_clean'].values]
word_count = [len(n.split(' ')) for n in messages]
import matplotlib.pyplot as plt

pd.DataFrame([len(n.split(' ')) for n in messages]).value_counts(normalize=True)

# what percentile of word_count is N?
import numpy as np
test['word_count'] = word_count
start = 5
end = 10
num = 123
[print('\n', n) for n in test[(test['word_count'] >start) & (test['word_count'] <end)]['message_with_interaction'].sample(n=20, random_state=num).values]
[print('\n', n) for n in test[(test['word_count'] >start) & (test['word_count'] <end)]['message_clean'].sample(n=20, random_state=num).values]
from scipy.stats import percentileofscore
percentileofscore(word_count,end, kind='mean')
plt.hist(word_count, bins=100)
plt.show()
test[(test['word_count'] >start) & (test['word_count'] <end)][ctl_tags13].sum() # still labelled 
# Remove below 10 words
end = 10
test[(test['word_count'] <end)]
test = test[(test['word_count'] >=end)]
test.shape

# Word count with interaction

word_count_wi = [len(n.split(' ')) for n in test['message_with_interaction'].values]
test['word_count_with_interaction'] = word_count_wi
plt.hist(word_count_wi, bins=1000)
# plot vertical line
plt.vlines(x=10, ymin=0, ymax=1000, color='red')
plt.show()
percentileofscore(word_count_wi,5, kind='mean')
start = 0
end = 20
test[(test['word_count_with_interaction'] >start) & (test['word_count_with_interaction'] <end)]
test.shape
test.to_parquet(input_dir + f'{set_name}_metadata_messages_clean.gzip', engine='pyarrow', compression='gzip', index = False)
# TODO: histograms
# TODO: do for training and testing



