import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

input_dir = './data/output/ml_performance/'

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

dirs = os.listdir(input_dir)

dirs_to_analyze = ['google-gemma-7b-it_2024-03-01T02-03-13_600', 
				   'google-gemma-2b-it_2024-03-01T16-56-22_600',
				#    'google-gemma-7b-it_2024-03-01T01-09-33_150',
				#    'google-gemma-2b-it_2024-03-01T01-02-33_60', 
				   
				   ]


# for dir in dirs:
# 	print(dir)
# 	print()
# 	os.listdir(input_dir+dir)


columns_to_keep = ['Model','Construct', 'Sensitivity', 'Specificity', 'Precision',
	   'F1', 'ROC AUC', 'Support',
	#    'PR AUC', 'Best th PR AUC', 
	   ]



def results_df(dvs, path_to_dir, columns_to_keep = 'all'):
	results_all = []
	for dv in dvs:
		files_dv = os.listdir(path_to_dir+'/'+dv+'/')
		try: results_file = [n for n in files_dv if 'results_' in n][0]
		except:
			print(dv, 'no results file, skipping')
			continue
		
		results_i = pd.read_csv(path_to_dir+'/'+dv+'/'+results_file, index_col = 0)

		cr_file = [n for n in files_dv if 'cr_' in n][0]
		cr_i = pd.read_csv(path_to_dir+'/'+dv+'/'+cr_file, index_col = 0)
		# display(cr_i)
		support = int(cr_i.loc['support'][0]+cr_i.loc['support'][1])

		
		results_i['Construct'] = prompt_names.get(dv).capitalize()
		results_i['Support'] = support

		results_all.append(results_i)

	

	results_all = pd.concat(results_all).round(2)
	results_all = results_all.reset_index(drop=True)
	results_all['Model'] = ['-'.join(n.split('-')[:-1]) for n in results_all['Model'].tolist()] #results_all['Model'] = [n.split('-')[:-1] for n in results_all['Model'].tolist()]

	if columns_to_keep != 'all':
		results_all = results_all[columns_to_keep]
	return results_all


def add_summary_row(df):
	summary_row_mean = {}
	summary_row_median = {}
	for column in df.columns:
		# if pd.api.types.is_numeric_dtype(df[column]):
		if pd.api.types.is_float_dtype(df[column]):
			mean_proportion = df[column].mean()
			median_proportion = df[column].median()
			min_proportion = df[column].min()
			max_proportion = df[column].max()
			summary_row_mean[column] = f'{mean_proportion:.2f} [{min_proportion:.2f} - {max_proportion:.2f}]'
			summary_row_median[column] = f'{median_proportion:.2f} [{min_proportion:.2f} - {max_proportion:.2f}]'
		else:
			summary_row_mean[column] = ''
			summary_row_median[column] = ''
	
	summary_df = pd.DataFrame([summary_row_mean, summary_row_median], index=['Mean [min-max]', 'Median [min-max]']).round(2)
	df_with_summary = pd.concat([df, summary_df], ignore_index=False)
	return df_with_summary


try: dirs.remove('.DS_Store')
except: pass
for dir in dirs_to_analyze:
	print('=========')
	print(dir)
	ctl_tags13_exist = [n for n in ctl_tags13 if n in os.listdir(input_dir+dir+'/')]
	# ctl_tags13_exist.remove('isolated')

	results_all = results_df(ctl_tags13_exist, input_dir+dir, columns_to_keep=columns_to_keep)	
	
	results_all = add_summary_row(results_all)
	display(results_all)
	results_all.to_csv(input_dir+dir+f'/results_all_constructs_{dir}.csv', index=False)
	


	