#!/usr/bin/env python
# coding: utf-8

# # Re-analysis of Original Experiments to Evaluate Randomness of Video Recommendation Choices

# # Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats 
import json
import re
import os, glob
from collections import Counter
from statistics import mode
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datetime import datetime
import gc
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
import math
from tqdm import tqdm
from collections import defaultdict
import random

import warnings
warnings.filterwarnings("ignore")

print('=' * 80 + '\n\n' + 'OUTPUT FROM: supplemental/thumbnails (first impressions)/13_thumbnail_null_comparison.py' + '\n\n')

with open("../../data/platform session data/sessions.json") as json_file:
	json_data = json.load(json_file)

# Actual ("Ground Truth") Labels
gun_videos_all_metadata = pd.read_csv('../../data/supplemental/metadata and ratings/metadata_w_label_June_2021_NLversion.csv')
wage_videos_all_metadata = pd.read_csv('../../data/supplemental/metadata and ratings/metadata_with_lables_binary_only_checked_0410.csv')
gun_labels = gun_videos_all_metadata[['originID', 'originCat']].dropna().drop_duplicates().rename(columns={"originID": "originId"})
wage_labels = wage_videos_all_metadata[['originID', 'originCat']].dropna().drop_duplicates().rename(columns={"originID": "originId"})
gold_labels = pd.concat([gun_labels, wage_labels], axis = 0)

# functions to get the original trees
def explore_branches(tree_df, all_trees_by_ID, all_trees_by_channelID, row_index = 0, step = 0):
	parent = tree_df.iloc[row_index]["originId"] # start with the first row
	channel_id = tree_df.iloc[row_index]["originChannelId"]
	
	# create set of unique keys per step
	if step not in all_trees_by_ID.keys():
		all_trees_by_ID[step] = set() 
	if step not in all_trees_by_channelID.keys():
		all_trees_by_channelID[step] = set() 
	
	all_trees_by_ID[step].add(parent) # add the parent's video ID to the relevant step
	all_trees_by_channelID[step].add(channel_id) # also store the channel ID
	
	for i in range(1, 4+1): # 4 + 1 because range() only prints up to n-1
		child_node = tree_df.iloc[row_index]["rec"+str(i)]

		# break if we hit a cycle
		if(child_node in set().union(*all_trees_by_ID.values())):
			break
		else:
			child_row_index = tree_df.index[tree_df['originId'] == child_node].tolist()
			explore_branches(tree_df, all_trees_by_ID, all_trees_by_channelID, child_row_index[0], step+1) # call recursively to get all the tree levels

def read_all_trees(tree_files):

	all_trees_by_ID = {}
	all_trees_by_channelID = {}
	all_tree_files_df = pd.DataFrame()

	for tree in tree_files:
		print(tree)

		# populate the tree
		tree_df = pd.read_csv(tree)
		
		explore_branches(tree_df, all_trees_by_ID, all_trees_by_channelID) # recusrively parse out video ID's and channel ID's from the trees.

		# save the dataframe to all_tree_files_df
		if(all_tree_files_df.empty):
			all_tree_files_df = tree_df
		else:
			all_tree_files_df = pd.concat([all_tree_files_df, tree_df], axis=0)

	return all_trees_by_ID, all_trees_by_channelID


tree_files_wage = glob.glob(os.path.join('../../data/recommendation trees/trees_wage/', '*.csv'))
tree_files_gun = glob.glob(os.path.join('../../data/recommendation trees/trees_gun/', '*.csv'))


all_trees_by_ID_wage, all_trees_by_channelID_wage = read_all_trees(tree_files_wage)
all_trees_by_ID_gun, all_trees_by_channelID_gun = read_all_trees(tree_files_gun)

# get the thing in parentheses as the topicid
gun_topicids = [re.search(r'\((.*?)\)', filename).group(1) for filename in tree_files_gun]
wage_topicids = [re.search(r'\((.*?)\)', filename).group(1) for filename in tree_files_wage]

# Do a little filtering to get the set of "real" participants
issue1 = pd.read_csv("../results/intermediate data/gun control (issue 1)/guncontrol_qualtrics_w123_clean.csv", dtype = {"urlid": str})

print("(STUDY 1) Full length of data:")
print(len(issue1))

issue1 = issue1.dropna(subset=["treatment_arm"])

print("number of treatment arm workers in complete data:")
print(len(issue1))

print("number once zero engagement is dropped:")
issue1 = issue1.dropna(subset=["pro", "anti"])
print(len(issue1))

print("number of unique ID's:")
print(len(issue1["worker_id"].drop_duplicates()))
# identify the duplicates from the original issue1
duplicate_workers_issue1 = issue1[issue1.duplicated(subset=["worker_id"], keep=False)]["worker_id"].drop_duplicates()
# keep only the first response per unique worker_id
issue1 = issue1.drop_duplicates(subset=["worker_id"], keep='first')


print("number once NA topicID/urlID are dropped:")
issue1 = issue1.dropna(subset=["topic_id", "urlid"])[["worker_id", "topic_id", "urlid"]]
print(len(issue1))

# merge in thirds
thirds_workerid_i1 = pd.read_csv("../results/intermediate data/gun control (issue 1)/guncontrol_qualtrics_w12_clean.csv")[["thirds", "worker_id"]].drop_duplicates()
issue1 = pd.merge(thirds_workerid_i1, issue1, on = "worker_id", how = "inner").drop_duplicates()

print("number once we merge in thirds from wave 2 (they are not present in the dataframe):")
print(len(issue1))

# take a look at the responses of the repeated workers
issue1_full = pd.read_csv("../results/intermediate data/gun control (issue 1)/guncontrol_qualtrics_w123_clean.csv", dtype = {"urlid": str})

for worker in duplicate_workers_issue1:
	worker_df = issue1_full[issue1_full["worker_id"]==worker][["gun_index_w3", "treatment_seed", "topic_id", "urlid"]].dropna()
	if worker_df.empty:
		continue
	mismatched_columns = []
	for col in worker_df.columns:
		if not worker_df[col].eq(worker_df[col].iloc[0]).all():
			mismatched_columns.append(col)
			print(f"Worker: {worker};  Cols with mismatches: {mismatched_columns}")

issue2 = pd.read_csv("../results/intermediate data/minimum wage (issue 2)/qualtrics_w12_clean.csv")

print("(STUDY 2) Full length of data:")
print(len(issue2))

issue2 = issue2.dropna(subset=["treatment_arm"])

print("number of treatment arm workers in complete data:")
print(len(issue2))

print("number once zero engagement is dropped:")
issue2 = issue2.dropna(subset=["pro", "anti"])
print(len(issue2))

print("number of unique ID's:")
print(len(issue2["worker_id"].drop_duplicates()))
# identify the duplicates from the original issue2
duplicate_workers_issue2 = issue2[issue2.duplicated(subset=["worker_id"], keep=False)]["worker_id"].drop_duplicates()
# keep only the first response per unique worker_id
issue2 = issue2.drop_duplicates(subset=["worker_id"], keep='first')

print("number once NA topicID/urlID are dropped:")
issue2 = issue2.dropna(subset=["topic_id", "urlid"])[["worker_id", "topic_id", "urlid", "thirds"]]
print(len(issue2))

# take a look at the responses of the repeated workers
issue2_full = pd.read_csv("../results/intermediate data/minimum wage (issue 2)/qualtrics_w12_clean.csv")

for worker in duplicate_workers_issue2:
	worker_df = issue2_full[issue2_full["worker_id"]==worker][["mw_support_w2", "treatment_seed", "topic_id", "urlid"]].dropna()
	if worker_df.empty:
		continue
	mismatched_columns = []
	for col in worker_df.columns:
		if not worker_df[col].eq(worker_df[col].iloc[0]).all():
			mismatched_columns.append(col)
			print(f"Worker: {worker};  Cols with mismatches: {mismatched_columns}")

yougov_topicids = pd.read_csv("../results/intermediate data/minimum wage (issue 2)/yg_w12_clean.csv")

print("(STUDY 3) Full length of data:")
print(len(yougov_topicids))

yougov_topicids = yougov_topicids.dropna(subset=["treatment_arm"])

print("number of treatment arm workers in complete data:")
print(len(yougov_topicids))

print("number once zero engagement is dropped:")
yougov_topicids = yougov_topicids.dropna(subset=["pro", "anti"])
print(len(yougov_topicids))

print("number of unique ID's:")
print(len(yougov_topicids["caseid"].drop_duplicates()))
# identify the duplicates from the original issue2
duplicate_workers_yougov = yougov_topicids[yougov_topicids.duplicated(subset=["caseid"], keep=False)]["caseid"].drop_duplicates()
# keep only the first response per unique worker_id
yougov_topicids = yougov_topicids.drop_duplicates(subset=["caseid"], keep='first')

print("number once NA topicID/urlID are dropped:")
yougov_topicids = yougov_topicids.dropna(subset=["topic_id", "urlid"])[["caseid", "topic_id", "urlid", "thirds"]]
print(len(yougov_topicids))

# Get the topicIds assigned for people who were partisan (aka, filter out moderates)

# these are the real people!

# rename "worker_id" and "caseid" to "id"
issue1 = issue1.rename(columns={"worker_id":"id"})
issue2 = issue2.rename(columns={"worker_id":"id"})
yougov_topicids = yougov_topicids.rename(columns={"caseid":"id"})

# Decision (10/30): Drop Issue 1 entirely, because some people were recommended videos that were not pro/anti, and we don't have access to the rec set
ALL_PARTICIPANTS = pd.concat([issue2[["id", "thirds", "urlid", "topic_id"]], yougov_topicids[["id", "thirds", "urlid", "topic_id"]]], axis = 0)

len(ALL_PARTICIPANTS)


# Parse out the condition (pro or anti) and the distribution (3-1 or 2-2) from the names and save them to the JSONs

# Function to parse out the condition (pro or anti) and the distribution (3-1 or 2-2) from the names
def parse_condition(topicId):
	topic_components = topicId.split("_")[-4:]
	distr = topic_components[1]
	political_leaning = 'anti' if 'a' in topic_components[-1] else 'pro'

	try:
		return (int(distr), political_leaning)
	except ValueError:
		print("unable to extract distribution for: " + str(topicId))
		# these are cases in which we can't extract a distribution; e.g., aTPMXi4EaKE_june2021_1_p
		# exclude them from analysis for now by returning None
		return (None, political_leaning)


# Function to turn a video ID into a political leaning

# this function converts from the longer video ID's (which have added numbers) to the "raw" format used in the gold labels
def convert_vid_to_political_leaning(vidId):
	# Keep up to the last non-numeric character and one trailing digit
	vidId_mod = re.match(r'(.*\D\d)\d*$', vidId)
	if vidId_mod:
		vidId = vidId_mod.group(1)

	matches = gold_labels[gold_labels["originId"] == vidId]["originCat"]

	if not matches.empty:
		return ', '.join(matches.astype(str))
	else:
		# If matches are empty, strip the last digit and try again
		vidId_mod = re.match(r'(.*\D)\d$', vidId)
		if vidId_mod:
			vidId = vidId_mod.group(1)
			matches = gold_labels[gold_labels["originId"] == vidId]["originCat"]
			if not matches.empty:
				return ', '.join(matches.astype(str))

	return None


# Filter the JSON blob to the LONGEST blob for each person

topic_urlid_to_blob_map = defaultdict(list)
for data_obj in json_data:
	key = (str(data_obj["topicID"]), str(data_obj["urlid"]))
	topic_urlid_to_blob_map[key].append(data_obj)
	
# get the longest data_obj per key
topic_urlid_to_max_blob_map = defaultdict(dict)
for key in topic_urlid_to_blob_map.keys():
	blobs_list = []
	vid_watch_times = []
	for blob in topic_urlid_to_blob_map[key]:
		try:
			vid_watch_time = np.sum(list(blob['vidWatchTimes'].values()))
		except KeyError:
			vid_watch_time = 0
		
		blobs_list.append(blob) 
		vid_watch_times.append(vid_watch_time)

	max_blob = blobs_list[np.argmax(vid_watch_times)]
	topic_urlid_to_max_blob_map[key] = max_blob


# This is our main function for reading in the recommendations & what people chose

def get_results_distribution(participant_url_identifiers, complete_only = False):

	print("Number of participant identifiers:")
	print(len(participant_url_identifiers))
	
	# create a set that appends the urlid to the topicId
	urlid_topicid_set = {f"{row['urlid']}_{row['topic_id']}" for _, row in participant_url_identifiers.iterrows()}

	# here, we're looking only at the MAX time blob for each person
	orig_experiment_json = [obj for obj in topic_urlid_to_max_blob_map.values() if (obj['urlid'] + '_' + obj['topicID'] in urlid_topicid_set)]

	# filter to only COMPLETE sessions
	if(complete_only == True):
		orig_experiment_json = [obj for obj in orig_experiment_json if obj["sessionFinished"] == True]

	print("Number of participant JSON objects:")
	print(len(orig_experiment_json))

	# unpack information about the distribution and condition from the topicId
	for obj in orig_experiment_json:
		obj['distribution'], obj['political_leaning'] = parse_condition(obj["topicID"])

	RESULTS_DICT = { # this is our main results dictionary
		'pro': {22:[], 31: []},
		'anti': {22:[], 31: []}
	}

	failures = { # log failures or reasons why we couldn't process all the data
		"no_distribution": 0,
		"videos_missing": 0,
		"recs_incomplete": 0,
		"activity_incomplete": 0,
		"video_non_pro_anti": 0,
		"reversed_politics_for_study1": 0}

	NOBS_counted = 0

	url_topic_id_processed = set()

	for obj in orig_experiment_json:

		participant_id = obj['urlid']

		processed = False  # Keep track of whether we processed this participant
		if(obj['distribution']) is None or obj['distribution'] not in {22, 31}:
			failures["no_distribution"] += 1
			continue  # We don't know the distribution; we can't analyze this

		try:
			recommendations = obj['displayOrders']
		except KeyError:
			failures["activity_incomplete"] += 1

		if not processed:  # Only process recommendations if not already processed
			try:
				recs_keys = ['2-recs', '3-recs', '4-recs', '5-recs']
				rec_info = {}
				for i, key in enumerate(recs_keys):
					rec_list = recommendations[key]
					rec_list_leanings = [convert_vid_to_political_leaning(vid) for vid in rec_list]
					rec_info[i+1] = {  # We start recs at level 2
						"videos": rec_list,
						"political_leaning": rec_list_leanings
					}
			except KeyError as e:
				failures["recs_incomplete"] += 1 # Save what we have and move on to the next participant

			watched_leanings = []
			try:
				p_watchlist = list(obj['vids'])
				for rec_level, video in enumerate(p_watchlist):
					if (rec_level == 0): continue  # Start recs at level 2
					if(video in rec_info[rec_level]["videos"]):
						political_leaning = convert_vid_to_political_leaning(video)
						if(political_leaning not in ['pro', 'anti']):
							continue
						watched_leanings.append(political_leaning)
			except KeyError as e:
				failures["activity_incomplete"] += 1
			
			RESULTS_DICT[obj['political_leaning']][obj['distribution']].append({participant_id: watched_leanings})
			processed = True  # Mark this participant as processed

		# Counts the number of participants for which we were able to get valid data
		if processed and (obj['topicID'], obj['urlid']) not in url_topic_id_processed:
			NOBS_counted += 1
			url_topic_id_processed.add((obj['topicID'], obj['urlid']))
	
	return RESULTS_DICT, NOBS_counted, failures, url_topic_id_processed


# This is the version that includes PARTIAL data

# run this for the liberals
print("Liberals:")
libs = ALL_PARTICIPANTS[ALL_PARTICIPANTS["thirds"] == 1]
results_lib, nobs_lib, failures_lib, url_topic_id_processed_lib = get_results_distribution(libs)

# run this for the conservatives
print("Conservatives:")
cons = ALL_PARTICIPANTS[ALL_PARTICIPANTS["thirds"] == 3]
results_cons, nobs_cons, failures_cons, url_topic_id_processed_cons = get_results_distribution(cons)

# (and just for kicks) run this for the moderates
print("Moderates:")
mods = ALL_PARTICIPANTS[ALL_PARTICIPANTS["thirds"] == 2]
results_mods, nobs_mods, failures_mods, url_topic_id_processed_mods = get_results_distribution(mods)

def turn_results_dict_to_dataframe(data):
    rows = []

    for top_level_key in data:
        for second_level_key in data[top_level_key]:
            for participant_dict in data[top_level_key][second_level_key]:
                for participant_id, choices in participant_dict.items():
                    row = {
                        'pro': 1 if top_level_key == 'pro' else 0,
                        'anti': 1 if top_level_key == 'anti' else 0,
                        '22': 1 if second_level_key == 22 else 0,
                        '31': 1 if second_level_key == 31 else 0,
                        'participantID': participant_id,
                        'choice_1': choices[0] if len(choices) > 0 else None,
                        'choice_2': choices[1] if len(choices) > 1 else None,
                        'choice_3': choices[2] if len(choices) > 2 else None,
                        'choice_4': choices[3] if len(choices) > 3 else None,
                    }
                    rows.append(row)

    return pd.DataFrame(rows)

df_results_lib = turn_results_dict_to_dataframe(results_lib)
df_results_cons = turn_results_dict_to_dataframe(results_cons)

# EXPERIMENT: drop anyone who didn't finish
df_results_lib_complete_only = df_results_lib.dropna(subset=["choice_1", "choice_2", "choice_3", "choice_4"])
df_results_cons_complete_only = df_results_cons.dropna(subset=["choice_1", "choice_2", "choice_3", "choice_4"])

# first, pivot the df to long
df_libs_long = pd.melt(
    df_results_lib,
    id_vars=['pro', 'anti', '22', '31', 'participantID'], 
    value_vars=['choice_1', 'choice_2', 'choice_3', 'choice_4'],  
    var_name='choice_number',
    value_name='choice'
).dropna(subset=["choice"])

df_cons_long = pd.melt(
    df_results_cons,
    id_vars=['pro', 'anti', '22', '31', 'participantID'], 
    value_vars=['choice_1', 'choice_2', 'choice_3', 'choice_4'],  
    var_name='choice_number',
    value_name='choice'
).dropna(subset=["choice"])

# also pivot the complete only dfs
df_libs_long_complete_only = pd.melt(
    df_results_lib_complete_only,
    id_vars=['pro', 'anti', '22', '31', 'participantID'], 
    value_vars=['choice_1', 'choice_2', 'choice_3', 'choice_4'],  
    var_name='choice_number',
    value_name='choice'
).dropna(subset=["choice"])

df_cons_long_complete_only = pd.melt(
    df_results_cons_complete_only,
    id_vars=['pro', 'anti', '22', '31', 'participantID'], 
    value_vars=['choice_1', 'choice_2', 'choice_3', 'choice_4'],  
    var_name='choice_number',
    value_name='choice'
).dropna(subset=["choice"])

# for lib, set 'pro' to 1 and 'anti' to 0 in choice_1, choice_2, choice_3, and choice_4
df_libs_long[["choice"]] = df_libs_long[["choice"]].applymap(lambda x: 1 if x == 'pro' else 0)
df_libs_long_complete_only[["choice"]] = df_libs_long_complete_only[["choice"]].applymap(lambda x: 1 if x == 'pro' else 0)

# for cons, set 'pro' to 0 and 'anti' to 1 in choice_1, choice_2, choice_3, and choice_4
df_cons_long[["choice"]] = df_cons_long[["choice"]].applymap(lambda x: 1 if x == 'anti' else 0)
df_cons_long_complete_only[["choice"]] = df_cons_long_complete_only[["choice"]].applymap(lambda x: 1 if x == 'anti' else 0)

df_cons_long_22 = df_cons_long[df_cons_long["22"] == 1]
df_cons_long_31 = df_cons_long[df_cons_long["31"] == 1]

# complete_only
df_cons_long_complete_only_22 = df_cons_long_complete_only[df_cons_long_complete_only["22"] == 1]
df_cons_long_complete_only_31 = df_cons_long_complete_only[df_cons_long_complete_only["31"] == 1]

df_libs_long_22 = df_libs_long[df_libs_long["22"] == 1]
df_libs_long_31 = df_libs_long[df_libs_long["31"] == 1]

# complete_only
df_libs_long_complete_only_22 = df_libs_long_complete_only[df_libs_long_complete_only["22"] == 1]
df_libs_long_complete_only_31 = df_libs_long_complete_only[df_libs_long_complete_only["31"] == 1]


# Linear Regression clustering by participant

# run the regression for conservatives in 22
y_cons22 = df_cons_long_22["choice"].astype(float)
X_cons22 = sm.add_constant(pd.Series(1, index=y_cons22.index))
model_cons22 = sm.OLS(y_cons22, X_cons22)
results_cons22 = model_cons22.fit(cov_type='cluster', cov_kwds={'groups': df_cons_long_22['participantID']})
print(results_cons22.summary())

# run the regression for conservatives in 31
y_cons31 = df_cons_long_31["choice"].astype(float)
X_cons31 = sm.add_constant(pd.Series(1, index=y_cons31.index))
model_cons31 = sm.OLS(y_cons31, X_cons31)
results_cons31 = model_cons31.fit(cov_type='cluster', cov_kwds={'groups': df_cons_long_31['participantID']})
print(results_cons31.summary())

# run the regression for liberals in 22
y_libs22 = df_libs_long_22["choice"].astype(float)
X_libs22 = sm.add_constant(pd.Series(1, index=y_libs22.index))
model_libs22 = sm.OLS(y_libs22, X_libs22)
results_libs22 = model_libs22.fit(cov_type='cluster', cov_kwds={'groups': df_libs_long_22['participantID']})
print(results_libs22.summary())

# run the regression for liberals in 31
y_libs31 = df_libs_long_31["choice"].astype(float)
X_libs31 = sm.add_constant(pd.Series(1, index=y_libs31.index))
model_libs31 = sm.OLS(y_libs31, X_libs31)
results_libs31 = model_libs31.fit(cov_type='cluster', cov_kwds={'groups': df_libs_long_31['participantID']})
print(results_libs31.summary())

# now run the regressions for the complete only dfs
# run the regression for conservatives in 22
y_cons22_complete_only = df_cons_long_complete_only_22["choice"].astype(float)
X_cons22_complete_only = sm.add_constant(pd.Series(1, index=y_cons22_complete_only.index))
model_cons22_complete_only = sm.OLS(y_cons22_complete_only, X_cons22_complete_only)
results_cons22_complete_only = model_cons22_complete_only.fit(cov_type='cluster', cov_kwds={'groups': df_cons_long_complete_only_22['participantID']})
print(results_cons22_complete_only.summary())

# run the regression for conservatives in 31
y_cons31_complete_only = df_cons_long_complete_only_31["choice"].astype(float)
X_cons31_complete_only = sm.add_constant(pd.Series(1, index=y_cons31_complete_only.index))
model_cons31_complete_only = sm.OLS(y_cons31_complete_only, X_cons31_complete_only)
results_cons31_complete_only = model_cons31_complete_only.fit(cov_type='cluster', cov_kwds={'groups': df_cons_long_complete_only_31['participantID']})
print(results_cons31_complete_only.summary())

df_libs_long_complete_only_22["choice"].dropna().mean().mean()

df_libs_long_complete_only_22["choice"].describe()

# complete only dfs
# run the regression for liberals in 22
y_libs22_complete_only = df_libs_long_complete_only_22["choice"].astype(float)
X_libs22_complete_only = sm.add_constant(pd.Series(1, index=y_libs22_complete_only.index))
model_libs22_complete_only = sm.OLS(y_libs22_complete_only, X_libs22_complete_only)
results_libs22_complete_only = model_libs22_complete_only.fit(cov_type='cluster', cov_kwds={'groups': df_libs_long_complete_only_22['participantID']})
print(results_libs22_complete_only.summary())

# run the regression for liberals in 31
y_libs31_complete_only = df_libs_long_complete_only_31["choice"].astype(float)
X_libs31_complete_only = sm.add_constant(pd.Series(1, index=y_libs31_complete_only.index))
model_libs31_complete_only = sm.OLS(y_libs31_complete_only, X_libs31_complete_only)
results_libs31_complete_only = model_libs31_complete_only.fit(cov_type='cluster', cov_kwds={'groups': df_libs_long_complete_only_31['participantID']})
print(results_libs31_complete_only.summary())


# ### Get stats for how many observations of each seed we processed

# Including INCOMPLETE data

# how many observations did we get valid data from?
print("Libs, Cons, Mods")
print(nobs_lib, nobs_cons, nobs_mods)
print("total:")
print(nobs_lib + nobs_cons + nobs_mods)

# ### Accounting by Topic ID + URL ID

# of those observations, how many UNIQUE topicID + urlids did we get?
### INCOMPLETE data included
print("Libs, Cons, Mods")
print(len(url_topic_id_processed_lib), len(url_topic_id_processed_cons), len(url_topic_id_processed_mods))
print("total:")
print(len(url_topic_id_processed_lib) + len(url_topic_id_processed_cons) +len(url_topic_id_processed_mods))


# ### Look at Failures
failures_lib, failures_cons, failures_mods

# ### Summary Statistics
def flatten_nested_dict(input_dict):
    result = {
        'pro': {22: [], 31: []},
        'anti': {22: [], 31: []}
    }
    
    for top_level_key, inner_dict in input_dict.items():
        for second_level_key, participants in inner_dict.items():
            for participant_dict in participants:
                for choices in participant_dict.values():
                    result[top_level_key][second_level_key].extend(choices)
    
    return result

results_lib_flat = flatten_nested_dict(results_lib)
results_cons_flat = flatten_nested_dict(results_cons)

len(results_cons_flat['anti'][22])
len(results_cons_flat['anti'][31])

def print_summary_stats_for_results(RESULTS_DICT):
	for seed, sub_dict in RESULTS_DICT.items():
		print(f"Summary statistics for '{seed}' seed:")
		
		for key, labels in sub_dict.items():
			total_labels = len(labels)
			if total_labels == 0:
				print(f"  List {key}: No labels to evaluate")
				continue
			
			# Count how many labels agree with the parent category ('pro' or 'anti')
			count_agree = sum(1 for label in labels if label == seed)
			percent_agree = (count_agree / total_labels) * 100
			
			# Print the statistics
			print(f"  [{key}]: {percent_agree:.2f}% selected videos with same partisanship as seed")
			
print("Liberals---------------------------")
print_summary_stats_for_results(results_lib_flat)
print("Conservatives----------------------")
print_summary_stats_for_results(results_cons_flat)


# ## Simulation of Random Guessing
# 
# This creates the baseline for which we compare everything

def simulate_random_watching(N_ITER = 1000000, video_set = [1, 0, 0, 0]): # 1 is pro and 0 is anti

    video_set_cur = video_set
    all_watched_videos = []

    for iter in range(N_ITER):
    
        watched_videos = []

        for i in range(4): # we make 4 choices

            # draws from the initial video distribution
            random_index = np.random.choice(4, 1)[0]
            random_video = video_set_cur[random_index]

            if(random_video == 1):
                video_set_cur = [1, 1, 1, 0]
            else:
                video_set_cur = [0, 0, 0, 1]

            watched_videos.append(random_video)
        
        all_watched_videos.append(watched_videos)
        # reset the video set
        video_set_cur = video_set
    
    return all_watched_videos

simulated_31_anti = simulate_random_watching(video_set=[1, 0, 0, 0])
simulated_31_anti_flat =  [item for sublist in simulated_31_anti for item in sublist]
print("Probability of selecting a pro video (given 3-1 distribution):")
print(1-np.mean(simulated_31_anti_flat))
p_31_anti = 1-np.mean(simulated_31_anti_flat)

simulated_31_pro = simulate_random_watching(video_set=[0, 1, 1, 1])
simulated_31_pro_flat = [item for sublist in simulated_31_pro for item in sublist]
print("Probability of selecting a pro video (given 3-1 distribution):")
print(np.mean(simulated_31_pro_flat))
p_31_pro = np.mean(simulated_31_pro_flat)

simulated_22 = simulate_random_watching(video_set=[1, 1, 0, 0])
simulated_22_flat =  [item for sublist in simulated_22 for item in sublist]
print("Probability of selecting a pro video (given 2-2 distribution):")
print(np.mean(simulated_22_flat))
p_22_pro = np.mean(simulated_22_flat)

print("Probability of selecting an anti video (given 2-2 distribution):")
print(1-np.mean(simulated_22_flat))
p_22_anti = 1-np.mean(simulated_22_flat)


# ### Statistical Tests

# - whether conservative respondents, given a current conservative video, clicked a conservative recommendation at >.75 rate in the 3/1, or .5 in the 2/2
# - whether conservative respondents, given a current liberal video, clicked a conservative recommendation at >.25 rate in the 3/1, or .5 in the 2/2
# - whether liberal respondents, given a current liberal video, clicked a liberal recommendation at >.75 rate in the 3/1, or .5 in the 2/2
# - whether liberal respondents, given a current conservative video, clicked a liberal recommendation at >.25 rate in the 3/1, or .5 in the 2/2

simulated_baselines = {
    "pro": {"22": p_22_pro, "31": p_31_pro},
    "anti": {"22": p_22_anti, "31": p_31_anti}
}

# Function to calculate the proportion of 'pro' or 'anti' matches in the list
def calculate_proportion_matches(results_list, target_key):
	matches = [1 if value == target_key else 0 for value in results_list]
	return matches

# Collect the proportions and run t-tests for both keys (22 and 31)
def run_stat_tests(RESULTS_DICT):
	for parent_key, data in RESULTS_DICT.items():
		print(f"Testing for parent key: {parent_key}")
		
		# For key 22, test against 0.5
		if(len(data[22]) > 0):
			proportion_22 = calculate_proportion_matches(data[22], parent_key)
			t_statistic_22, p_value_22 = stats.ttest_1samp(a=proportion_22, popmean=simulated_baselines[parent_key]["22"])
			print(f"t-test for 22 key (test {np.mean(proportion_22)} against {simulated_baselines[parent_key]['22']})")
			print(f"t-statistic: {round(t_statistic_22, 5)}")
			print(f"p-value: {round(p_value_22, 5)}")
		
		# For key 31, test against 0.75
		if(len(data[31]) > 0):
			proportion_31 = calculate_proportion_matches(data[31], parent_key)
			t_statistic_31, p_value_31 = stats.ttest_1samp(a=proportion_31, popmean=simulated_baselines[parent_key]["31"])
			print(f"t-test for 31 key (test {np.mean(proportion_31)} against {simulated_baselines[parent_key]['31']})")
			print(f"t-statistic: {round(t_statistic_31, 10)}")
			print(f"p-value: {round(p_value_31, 5)}")

print("Liberals---------------------------")
run_stat_tests(results_lib_flat)
print("Conservatives----------------------")
run_stat_tests(results_cons_flat)


# # Figuring out data anomalies

# First, there are some overlapping participants (326) between Study 1 and Study 2

len(set(issue1["id"]).intersection(set(issue2["id"])))


# These are the topicID's for everyone we expect to have data for

ALL_PARTICIPANTS

# Search for the participant in the JSON data and figure out whether we have at least one session for the participant in which they actually viewed data

# these are the number of Topic/URL Id's among our participants
unique_participant_topic_urlid = set([(str(row["topic_id"]), str(row["urlid"])) for _, row in ALL_PARTICIPANTS.iterrows()])
len(unique_participant_topic_urlid)

# these are the number of Topic/URL Id's within the JSON data
unique_json_topic_urlid = set([(str(obj["topicID"]), str(obj["urlid"])) for obj in json_data])
len(unique_json_topic_urlid)

# Actually, every participant is in the data somewhere!
len(unique_json_topic_urlid.intersection(unique_participant_topic_urlid))


# Figure out what each of the participants did on the platform
participant_session_counts = defaultdict(int)
participant_vids_present_session_counts = defaultdict(int)
participant_complete_session_counts = defaultdict(int)

participants_set = set()

topic_urlid_data_map = defaultdict(list)

for data_obj in json_data:
	key = (str(data_obj["topicID"]), str(data_obj["urlid"]))
	topic_urlid_data_map[key].append({
		"has_vids": 'vids' in data_obj.keys(),
		"sessionFinished": data_obj.get("sessionFinished", False)
	})

for _, participant in tqdm(ALL_PARTICIPANTS.iterrows(), total=len(ALL_PARTICIPANTS), desc="Processing participants"):
	
	topicID = str(participant["topic_id"])
	urlid = str(participant["urlid"])
	participant_id = str(participant["id"])

	# Use the (topicID, urlid) pair as the key
	key = (topicID, urlid)

	# Get the relevant data objects from the pre-built map
	if key in topic_urlid_data_map:
		
		participants_set.add(participant_id)

		for data_obj in topic_urlid_data_map[key]:
			# Track if the participant has any appearances in the data at all
			participant_session_counts[key] += 1

			# Track how many times the participant saw valid videos
			if data_obj["has_vids"]:
				participant_vids_present_session_counts[key] += 1

			# Track how many times they have complete data
			if data_obj["sessionFinished"]:
				participant_complete_session_counts[key] += 1


# Who actually completed their sessions?
len(participant_session_counts) # everyone is in the data somewhere

len(participant_vids_present_session_counts) # everyone has some kind of video interaction

len(participant_complete_session_counts) # only 5,573 people fully completed the study

len(ALL_PARTICIPANTS)


# But why is the number of unique participant ID's different?
# 
# **It turns out that it's because of the overlap in participants between Study 1 and Study 2!!!**

# assert(len(ALL_PARTICIPANTS) - len(participants_set) == len(set(issue1["id"]).intersection(set(issue2["id"]))))


# If we look only at "complete" sessions: the same participant has up to 3 complete sessions
# 
# If we include incomplete sessions: the same participant can have up to 14 partial sessions (!!!)

pd.Series(participant_complete_session_counts.values()).value_counts()

pd.Series(participant_vids_present_session_counts.values()).value_counts()

pd.Series(participant_session_counts.values()).value_counts()

# Who are the participants who didn't complete the study, and what were their other survey DV's like?

no_session = unique_participant_topic_urlid.difference(set(participant_complete_session_counts.keys()))

len(no_session)


# 272 were in Study 1
study1_topic_urlid = set([(str(row["topic_id"]), str(row["urlid"])) for _, row in issue1.iterrows()])
len(study1_topic_urlid.intersection(no_session))

# 196 were in Study 2
study2_topic_urlid = set([(str(row["topic_id"]), str(row["urlid"])) for _, row in issue2.iterrows()])
len(study2_topic_urlid.intersection(no_session))


# 3 were in Study 3
yougov_topic_urlid = set([(str(row["topic_id"]), str(row["urlid"])) for _, row in yougov_topicids.iterrows()])
len(yougov_topic_urlid.intersection(no_session))

issue1_full = pd.read_csv("../results/intermediate data/gun control (issue 1)/guncontrol_qualtrics_w123_clean.csv")
issue2_full = pd.read_csv("../results/intermediate data/minimum wage (issue 2)/qualtrics_w12_clean.csv")
yougov_full = pd.read_csv("../results/intermediate data/minimum wage (issue 2)/yg_w12_clean.csv")

columns_to_collect = ["duration", "total_interactions", "gun_index_w2", "gun_index_2", "gun_index_w3", "stricter_laws_w3",
					  "right_to_own_importance_w3", "assault_ban_w3", "handgun_ban_w3", "concealed_safe_w3",
					  "gun_index_2_w3", "mw_index_w2", "trust_youtube_w2", "media_trust_w2", "media_trust_w3", "affpol_smart", 
					  "smart_dems_w2", "smart_reps_w2"]

collected_values = {col: [] for col in columns_to_collect}
num_redundancies_by_study = {"Study_1": 0, "Study_2": 0, "Study_3": 0}

def collect_values_from_study(df, id_column, participant_ids, collected_values):
	filtered_df = df[df[id_column].isin(participant_ids)]
	for col in columns_to_collect:
		if col in filtered_df.columns:
			collected_values[col].extend(filtered_df[col].dropna().tolist())

for (topic_id, urlid) in no_session:
	# Find matches in the datasets
	matches_i1 = issue1[(issue1['topic_id'] == topic_id) & (issue1['urlid'] == urlid)]
	matches_i2 = issue2[(issue2['topic_id'] == topic_id) & (issue2['urlid'] == urlid)]
	matches_yg = yougov_topicids[(yougov_topicids['topic_id'] == topic_id) & (yougov_topicids['urlid'] == urlid)]

	if not matches_i1.empty:
		num_redundancies_by_study["Study_1"] += 1
		collect_values_from_study(issue1_full, "worker_id", matches_i1["id"], collected_values)

	if not matches_i2.empty:
		num_redundancies_by_study["Study_2"] += 1
		collect_values_from_study(issue2_full, "worker_id", matches_i2["id"], collected_values)

	if not matches_yg.empty:
		num_redundancies_by_study["Study_3"] += 1
		collect_values_from_study(yougov_full, "caseid", matches_yg["id"], collected_values)

# Filter columns with non-empty collected values
non_empty_cols = {col: values for col, values in collected_values.items() if values}

# Set up the plot grid based on the number of columns with data
num_plots = len(non_empty_cols)
n_cols = 3  # Maximum number of columns per row
n_rows = math.ceil(num_plots / n_cols)  # Dynamically calculate number of rows

# Set up the figure size dynamically based on number of plots
plt.figure(figsize=(5 * n_cols, 4 * n_rows))

# Plot histograms for non-empty columns
for i, (col, values) in enumerate(non_empty_cols.items(), 1):
	plt.subplot(n_rows, n_cols, i)
	plt.hist(values, bins=20, color='blue', edgecolor='black')
	plt.title(col)
	plt.xlabel(col)
	plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

num_redundancies_by_study

