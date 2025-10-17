#!/usr/bin/env python
# coding: utf-8

# # THUMBNAILS EXPERIMENT

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

import warnings
warnings.filterwarnings("ignore")

print('=' * 80 + '\n\n' + 'OUTPUT FROM: supplemental/thumbnails (first impressions)/12_thumbnail_analysis.py' + '\n\n')

# Reading Session Logs

# Session logs --- pIDHash is our new respondent-id
with open("../../data/platform session data/sessions.json") as json_file:
    json_data = json.load(json_file)


unique_topics = [] 
real_data = []

for item in json_data:

    # Check if the session is completed
    if item['sessionFinished']:
    
        # Check if the topic ID is 'min_wage' or 'gun_control'
        if item['topicID'] in ['min_wage','gun_control']:

            # Convert start time from milliseconds to seconds
            unix_time_seconds_start = item['startTime'] / 1000
            
            # Convert the UNIX timestamp to a datetime object
            normal_time_start = datetime.fromtimestamp(unix_time_seconds_start)

            # Check if the session started on May 23, 2024, after 10 AM
            if (normal_time_start.year == 2024 and
                normal_time_start.month == 5 and
                normal_time_start.day == 23 and
                normal_time_start.hour >= 10):
                
                # Add the session to the real_data list
                real_data.append(item)


del json_data
gc.collect()


print('Total session count:', len(real_data))


# # GPT Ratings


# GPT Labels
gpt_labels = pd.read_csv('../../data/supplemental/metadata and ratings/gpt_thumb_ratings_withHumanInfo.csv').drop_duplicates()

# in case of duplicate labels by GPT, take the first one
gpt_labels = gpt_labels.drop_duplicates(subset='originId', keep='first')

gpt_labels['gpt_thumb_rating'] = gpt_labels['gpt_thumb_rating'].replace('pro.', 'pro') # 2 
gpt_labels['gpt_thumb_rating'] = gpt_labels['gpt_thumb_rating'].replace('anti.', 'anti')# 3
gpt_labels.gpt_thumb_rating.value_counts() # 1%


# # Gold Standard Labels

# these were the videos we actually provided to Jim
labels_on_platform = pd.concat([pd.read_csv('../../data/supplemental/metadata and ratings/gun_thumbnails_updated_v4(gun_control).csv'), pd.read_csv('../../data/supplemental/metadata and ratings/wage_thumbnails_updated_v4(min_wage).csv')])

# Actual ("Ground Truth") Labels
gun_videos_all_metadata = pd.read_csv('../../data/supplemental/metadata and ratings/metadata_w_label_June_2021_NLversion.csv')
wage_videos_all_metadata = pd.read_csv('../../data/supplemental/metadata and ratings/metadata_with_lables_binary_only_checked_0410.csv')
gun_labels = gun_videos_all_metadata[['originID', 'originCat']].dropna().drop_duplicates().rename(columns={"originID": "originId"})
wage_labels = wage_videos_all_metadata[['originID', 'originCat']].dropna().drop_duplicates().rename(columns={"originID": "originId"})
gold_labels = pd.concat([gun_labels, wage_labels], axis = 0)


# # Curate a Subset of "Easy to Rate" Videos

bert_ratings = pd.read_csv('../../data/supplemental/metadata and ratings/bert_rated_wage_videos_all.csv')

gpt_continuous_extremenss_ratings = pd.read_csv('../../data/supplemental/metadata and ratings/gpt_continuous_ratings_minwage.csv').drop_duplicates()
gpt_continuous_extremenss_ratings = gpt_continuous_extremenss_ratings.rename(columns={"naijia_originId": "originID"})
gpt_continuous_extremenss_ratings = gpt_continuous_extremenss_ratings[["originID", "gpt_label"]]

bert_ratings = bert_ratings.rename(columns={"naijia_originId": "originID"})

ratings_aggregated = pd.merge(gpt_continuous_extremenss_ratings, bert_ratings[["originID", "originCat", "bert_score"]], how= "inner", on = 'originID').drop_duplicates()

# convert gpt label to 'pro' if it is < 0 and 'anti' otherwise
ratings_aggregated['gpt_label'] = np.where(ratings_aggregated['gpt_label'] < 0, 'pro', 'anti')
# do the same thing for bert_score
ratings_aggregated['bert_score'] = np.where(ratings_aggregated['bert_score'] < 0, 'pro', 'anti')

# get cases where all 3 agree
ratings_aggregated['all_agree'] = np.where((ratings_aggregated['gpt_label'] == ratings_aggregated['bert_score']) & (ratings_aggregated['gpt_label'] == ratings_aggregated['originCat']), 1, 0)


# This is a subset of minimum wage videos that is "easy to rate": GPT, BERT, and the human Turkers all got this correct

MINWAGE_AGREED_VIDEOS = ratings_aggregated[ratings_aggregated['all_agree'] == 1]

# get cases where GPT and BERT agree, but humans (originCat) disagrees
ratings_aggregated['gpt_bert_agree'] = np.where((ratings_aggregated['gpt_label'] == ratings_aggregated['bert_score']) & (ratings_aggregated['gpt_label'] != ratings_aggregated['originCat']), 1, 0)

# peek at specific videos
gpt_continuous_extremenss_ratings[gpt_continuous_extremenss_ratings["originID"]=="Z_r5TlBdjEM"]

# this happened in just 7 out of 154 cases...
ratings_aggregated['gpt_bert_agree'].value_counts()

# .... or 4.5%!
ratings_aggregated['gpt_bert_agree'].value_counts()[1]/len(ratings_aggregated)

ratings_aggregated[ratings_aggregated['gpt_bert_agree']==1]


# # Session Level Performance I

# Including videos without GPT labels (both no label and 'insufficient information' label cases)

def session_rep_counts(data):
    exp_sessions_wage, exp_sessions_gun = [], []
    exp_indexes_wage, exp_indexes_gun = [], []
    exp_indexes_wage_resp, exp_indexes_gun_resp = [], []
    
    for i in range(0, len(data)): # Iterate over all elements in the dictionary
    
        # Only completed sessions for our surveys
        if data[i]['sessionFinished'] and data[i]['topicID'] in ['min_wage', 'gun_control'] and len(data[i]['ratingResults']) > 0:
    
            topic_id = data[i]['topicID']
            resp_id = data[i]['pIDHash']

            if data[i]['topicID'] == 'min_wage':
                exp_sessions_wage.append(topic_id)
                exp_indexes_wage_resp.append(resp_id)
                exp_indexes_wage.append(i)
        
            elif data[i]['topicID'] == 'gun_control':
                exp_sessions_gun.append(topic_id)
                exp_indexes_gun_resp.append(resp_id)
                exp_indexes_gun.append(i)
    
    print('Wage session count:',len(exp_indexes_wage))
    print('Gun session count:',len(exp_indexes_gun))
    
    print('Unique respondents (authID) in Wage:',len(np.unique(exp_indexes_wage_resp)))
    print('Unique respondents (authID) in Gun',len(np.unique(exp_indexes_gun_resp)))

    return exp_indexes_wage, exp_indexes_gun

def exp_analysis(data, index, total_count, pro_count, anti_count, insuf_count, nolabel_count, exc):

    gold_matches = []
    pro_matches = []
    anti_matches = []
    
    for item in data[index]['ratingResults']:
        
        total_count += 1
        exp_index = int(item['index'])

        if data[index]['topicID'] == 'min_wage':
            if exp_index == 1:
                exp_label = 'pro'
            elif exp_index == 2:
                exp_label = 'anti'
            elif exp_index == 3:
                exp_label = 'insufficient data.'
        elif data[index]['topicID'] == 'gun_control':
            if exp_index == 1:
                exp_label = 'anti'
            elif exp_index == 2:
                exp_label = 'pro'
            elif exp_index == 3:
                exp_label = 'insufficient data.'

        try:
            gpt_label = gpt_labels[gpt_labels.originId == item['vid']].gpt_thumb_rating.values[0]
        except:
            gpt_label = 'no_label'

        try:
            gold_label = gold_labels[gold_labels.originId == item['vid']].originCat.values[0]
        except:
            gold_label = 'no_label'
    
        if gold_label == 'pro':
            pro_count += 1
        elif gold_label == 'anti':
            anti_count += 1
        elif gold_label == 'insufficient data.':
            insuf_count += 1
        else:
            nolabel_count += 1
        
        if exc == 0:
            gold_matches.append(1 if exp_label == gold_label else 0)
            # determine whether it's an anti or pro match
            if exp_label == 'pro':
                pro_matches.append(1 if exp_label == gold_label else 0)
            elif exp_label == 'anti':
                anti_matches.append(1 if exp_label == gold_label else 0)
        elif exc == 1:
            if gpt_label == 'insufficient data.' or pd.isnull(gold_label):
                continue
            else:
                gold_matches.append(1 if exp_label == gold_label else 0)
                
    return gold_matches, pro_matches, anti_matches, pro_count, anti_count, insuf_count, nolabel_count, total_count

def results(total_count,insuf_count,nolabel_count,pro_count,anti_count,gold_matches):
    print('Number of videos:', total_count)
    print('Number of labeled videos:', total_count - insuf_count - nolabel_count)
    print('Number of pro videos:', pro_count)
    print('Number of anti videos:', anti_count)
    print('Number of vague videos:', insuf_count)
    print('Number of non labeled videos:', nolabel_count)
    print("***")
    print('Total number of matches with GPT:', np.sum(gold_matches))
    print("***")
    print('Total % of matches with GPT %', np.round(np.sum(gold_matches) / len(gold_matches), 2) * 100)
    print("***")
    print('')

# json_data is the json data
# exc 0 if we want to include videos without GPT labels, 1 otw.
def thumbnail_exp_check(data, exc=0):

    result_df = pd.DataFrame(columns = ['session_id',
                                        'topic_id',
                                        'respondent_id',
                                        'total_video_count',
                                        'respondent_label_count',
                                        'gold_insufficient_video_count',
                                        'gold_nolabel_video_count',
                                        'gold_pro_video_count',
                                        'gold_anti_video_count',
                                        'gold_match_count',
                                        'gold_pro_match_count',
                                        'gold_anti_match_count'
                                       ])

    print('Summary Statistics')
    exp_indexes_wage, exp_indexes_gun = session_rep_counts(data) # session and unique respondent counts

    # Check all matches for each experiment
    for index in exp_indexes_wage + exp_indexes_gun:

        if data[index]['sessionFinished']:

            if len(data[index]['ratingResults']) > 0:

                pro_count, anti_count, insuf_count, nolabel_count, total_count = 0, 0, 0, 0, 0
        
                gold_matches, pro_matches, anti_matches, pro_count, anti_count, insuf_count, nolabel_count, total_count = exp_analysis(data, index, total_count, pro_count, 
                                                                           anti_count, insuf_count, nolabel_count, 
                                                                           exc=exc)

                resp_id = data[index]['pIDHash']

                row = [index,
                       data[index]['topicID'],
                       resp_id, 
                       len(data[index]['ratingResults']),
                       total_count, 
                       insuf_count, 
                       nolabel_count, 
                       pro_count, 
                       anti_count, 
                       np.sum(gold_matches),
                       np.sum(pro_matches),
                       np.sum(anti_matches)
                      ]
                row_df = pd.DataFrame(row).T
                row_df.columns = ['session_id',
                                        'topic_id',
                                        'respondent_id',
                                        'total_video_count',
                                        'respondent_label_count',
                                        'gold_insufficient_video_count',
                                        'gold_nolabel_video_count',
                                        'gold_pro_video_count',
                                        'gold_anti_video_count',
                                        'gold_match_count',
                                        'gold_pro_match_count',
                                        'gold_anti_match_count'
                                       ]

                result_df = pd.concat([result_df, row_df], axis=0)

    result_df['gold_match_perc'] = result_df['gold_match_count'] / result_df['total_video_count']
    ## look at the specific breakdown of pro versus anti
    result_df['gold_pro_match_perc'] = result_df['gold_pro_match_count'] / result_df['gold_pro_video_count']
    result_df['gold_anti_match_perc'] = result_df['gold_anti_match_count'] / result_df['gold_anti_video_count']
    
    return result_df

# SESSION LEVEL RESULT
result_df = thumbnail_exp_check(real_data, exc=0)

print("number of unique participants")
len(result_df)

result_df["total_video_count"].value_counts()

def calculate_quartiles(series):
    rounded = []
    for i in series.quantile([0.25, 0.5, 0.75]).to_list():
        rounded.append(np.round(i,2))
    return rounded

result_df.groupby('topic_id').agg(
    session_count=('session_id', 'nunique'),
    respondent_count=('respondent_id', 'nunique'),
    gold_match_mean=('gold_match_perc', 'mean'),
    gold_match_std=('gold_match_perc', 'std'),
    gold_match_quartiles=('gold_match_perc', calculate_quartiles)
).reset_index()

# examine the results by pro/anti
results_by_pro_anti = result_df.groupby('topic_id').agg(
    session_count=('session_id', 'nunique'),
    respondent_count=('respondent_id', 'nunique'),
    pro_gold_match_mean=('gold_pro_match_perc', 'mean'),
    pro_gold_match_std=('gold_pro_match_perc', 'std'),
    pro_gold_match_quartiles=('gold_pro_match_perc', calculate_quartiles),
    anti_gold_match_mean=('gold_anti_match_perc', 'mean'),
    anti_gold_match_std=('gold_anti_match_perc', 'std'),
    anti_gold_match_quartiles=('gold_anti_match_perc', calculate_quartiles)
).reset_index()

melted_df = pd.melt(results_by_pro_anti, 
                    id_vars=['topic_id', 'session_count', 'respondent_count'], 
                    value_vars=[
                        'pro_gold_match_mean', 
                        'pro_gold_match_std', 
                        'pro_gold_match_quartiles', 
                        'anti_gold_match_mean', 
                        'anti_gold_match_std', 
                        'anti_gold_match_quartiles'],
                    var_name='stat_type', value_name='value')
melted_df['type'] = melted_df['stat_type'].apply(lambda x: x.split('_')[0])
melted_df['statistic'] = melted_df['stat_type'].apply(lambda x: x.split('_')[-1])
melted_df = melted_df.sort_values(by=['topic_id', 'statistic']).drop('stat_type', axis=1)
melted_df

print("t-test for gold match percentage (pooled)")
all_gold_match_perc = np.asarray([float(num) for num in result_df["gold_match_perc"]])
t_statistic, p_value = stats.ttest_1samp(a=all_gold_match_perc, popmean=0.5) 
print("t-statistic:", t_statistic)
print("p-value:", p_value)

print("overall accuracy")
print(np.mean(all_gold_match_perc))

print("t-test for gold match percentage (liberal gun control; operationalizing random as 1/3)")
gun_result_df = result_df[result_df["topic_id"] == "gun_control"]
gun_gold_match_perc = np.asarray([float(num) for num in gun_result_df["gold_pro_match_perc"]])
t_statistic, p_value = stats.ttest_1samp(a=gun_gold_match_perc, popmean=0.333) 
print("t-statistic:", t_statistic)
print("p-value:", p_value)

result_df_broken_by_proanti = result_df[["topic_id", "gold_pro_match_perc", "gold_anti_match_perc"]]
result_df_proanti_melted = pd.melt(result_df_broken_by_proanti, 
                    id_vars=['topic_id'], 
                    value_vars=[
                        'gold_pro_match_perc',
                        'gold_anti_match_perc'],
                    var_name='stat_type', value_name='value')
result_df_proanti_melted['Video Gold Label'] = result_df_proanti_melted['stat_type'].apply(lambda x: x.split('_')[1])
result_df_proanti_melted = result_df_proanti_melted.drop('stat_type', axis=1)

sns.boxplot(x='topic_id', y='gold_match_perc', data=result_df)
plt.show()

# make the labels more understandable
result_df_proanti_melted["Video Gold Label"]= result_df_proanti_melted["Video Gold Label"].replace({"pro": "Liberal", "anti": "Conservative"})
result_df_proanti_melted["topic_id"]= result_df_proanti_melted["topic_id"].replace({"min_wage": "Minimum Wage", "gun_control": "Gun Control"})

sns.boxplot(x='topic_id', y='value', hue='Video Gold Label', data=result_df_proanti_melted)
# add a horizontal line for the 50% mark
plt.axhline(0.5, color='r', linestyle='--')
plt.title("Individual Raters' Percentage Match by Topic and Gold Label")
plt.xlabel("Topic")
plt.ylabel("Percentage Match with Gold")
plt.show()


# # Video Level Performance
exp_indexes_wage, exp_indexes_gun = session_rep_counts(real_data)

indexes = exp_indexes_wage + exp_indexes_gun 
videos = {}
videos['gun_control'] = {}
videos['min_wage'] = {}

for index in indexes:
    
    ratings = real_data[index]['ratingResults']
    for rating in ratings:
        video = rating['vid']
        exp_index = int(rating['index'])

        if real_data[index]['topicID'] == 'min_wage':
            if exp_index == 1:
                exp_label = 'pro'
            elif exp_index == 2:
                exp_label = 'anti'
            elif exp_index == 3:
                exp_label = 'insufficient data.'
        elif real_data[index]['topicID'] == 'gun_control':
            if exp_index == 1:
                exp_label = 'anti'
            elif exp_index == 2:
                exp_label = 'pro'
            elif exp_index == 3:
                exp_label = 'insufficient data.'

        if real_data[index]['topicID'] == 'min_wage':
            if video not in videos['min_wage'].keys():
                videos['min_wage'][video] = [exp_label]
            else:
                videos['min_wage'][video].append(exp_label)
                
        elif real_data[index]['topicID'] == 'gun_control':
            if video not in videos['gun_control'].keys():
                videos['gun_control'][video] = [exp_label]
            else:
                videos['gun_control'][video].append(exp_label)

for video, labels in videos.items():
    print(video)

majority_votes = {}
vote_counts = []
topics = []
num_votes_for_majority = []
majority_vote_drop_insuf = []

for topicid, videolist in videos.items():

    for video, labels in videolist.items():
        majority_vote = mode(labels)
        vote_count = len(labels)
        votes_for_majority = [label for label in labels if label == majority_vote]
        num_votes_for_majority.append(len(votes_for_majority))

        # a version of majority_vote if we remove votes for "insufficient data."
        majority_vote_drop_insuf.append(mode([label for label in labels if label != 'insufficient data.']))

        # Debug what happens when we have a very small minority as the majority percentage
        # how can the mode have only something like 35%?

        # Answer -- it's because there's a 3-way split between 'pro,' 'anti,' and 'insufficient data.'

        # if len(votes_for_majority) / vote_count < 0.4:
        #     print('Majority vote:', majority_vote)
        #     print('Votes for majority:', len(votes_for_majority))
        #     print('Total votes:', vote_count)
        #     print('Votes for majority %:', len(votes_for_majority) / vote_count)
        #     print('Labels:')
        #     print(pd.Series(labels).value_counts())
        #     print('***')
        
        majority_votes[video] = majority_vote
        vote_counts.append(vote_count)
        topics.append(topicid)

majority_voting = pd.concat([pd.Series(majority_votes.keys()), 
                             pd.Series(topics), 
                             pd.Series(vote_counts), 
                             pd.Series(num_votes_for_majority),
                             pd.Series(majority_votes.values()),
                             pd.Series(majority_vote_drop_insuf)], 
                            axis=1)
majority_voting.columns = ['originId','topicID','vote_count', 'num_votes_for_majority', 'majority_label', 'majority_label_drop_insuf']
print(majority_voting.shape)

majority_voting = majority_voting.merge(gold_labels.drop_duplicates(), how='left',on='originId')
majority_voting = majority_voting.merge(gpt_labels.drop_duplicates(), how='left',on='originId')

majority_voting.head()
majority_voting.originCat.value_counts()
majority_voting.gpt_thumb_rating.value_counts()

# Encode the labels
majority_voting['gold_label_encoded'] = majority_voting['originCat'].map({'anti': 0, 
                                                                  'pro': 1, 
                                                                  'other': 2})

majority_voting['majority_label_encoded'] = majority_voting['majority_label'].map({'anti': 0, 
                                                                           'pro': 1, 
                                                                           'insufficient data.': 2})

majority_voting['majority_label_drop_insuf_encoded'] = majority_voting['majority_label_drop_insuf'].map({'anti': 0, 
                                                                           'pro': 1})

majority_voting['gpt_encoded'] = majority_voting['gpt_thumb_rating'].map({'anti': 0,
                                                                  'pro': 1,
                                                                  'insufficient data.': 2})

def performance_metrics(filtered_df, topicID='OVERALL'):
    print('**********')
    print(f'{topicID}')

    if topicID == 'OVERALL':
        accuracy = accuracy_score(filtered_df['gold_label_encoded'], 
                                  filtered_df['majority_label_encoded'])
        precision = precision_score(filtered_df['gold_label_encoded'], 
                                    filtered_df['majority_label_encoded'], 
                                    labels = [0,1,2],
                                    average='weighted',
                                    zero_division=0)
        recall = recall_score(filtered_df['gold_label_encoded'], 
                              filtered_df['majority_label_encoded'],
                              labels = [0,1,2], 
                              average='weighted',
                              zero_division=0)
    else:
        accuracy = accuracy_score(filtered_df[filtered_df.topicID == f'{topicID}']['gold_label_encoded'], 
                                  filtered_df[filtered_df.topicID == f'{topicID}']['majority_label_encoded'])
        precision = precision_score(filtered_df[filtered_df.topicID == f'{topicID}']['gold_label_encoded'], 
                                    filtered_df[filtered_df.topicID == f'{topicID}']['majority_label_encoded'],
                                    labels = [0,1,2], 
                                    average='weighted',
                                    zero_division=0)
        recall = recall_score(y_true = filtered_df[filtered_df.topicID == f'{topicID}']['gold_label_encoded'], 
                              y_pred = filtered_df[filtered_df.topicID == f'{topicID}']['majority_label_encoded'], 
                              labels = [0,1,2],
                              average= 'weighted',
                              zero_division=0)
                
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print('**********')

majority_voting_nonNullOriginCat = majority_voting[majority_voting.originCat.isnull() == False]

majority_voting_nonNullOriginCat["majority_label"].value_counts()

pro_only_maj_vote = majority_voting_nonNullOriginCat[majority_voting_nonNullOriginCat['originCat'] == 'pro']
anti_only_maj_vote = majority_voting_nonNullOriginCat[majority_voting_nonNullOriginCat['originCat'] == 'anti']
# flip all the labels for anti_only
anti_only_maj_vote = anti_only_maj_vote.copy()
anti_only_maj_vote.loc[:, 'majority_label_encoded'] = anti_only_maj_vote['majority_label_encoded'].replace({0: 1, 1: 0})
anti_only_maj_vote.loc[:, 'gold_label_encoded'] = anti_only_maj_vote['gold_label_encoded'].replace({0: 1, 1: 0})
anti_only_maj_vote.loc[:, 'majority_label_drop_insuf_encoded'] = anti_only_maj_vote['majority_label_drop_insuf_encoded'].replace({0: 1, 1: 0})

print("number of videos that had a valid original rating")
print(len(majority_voting_nonNullOriginCat))

majority_voting_nonNullOriginCat.loc[:, "is_human_match"] = majority_voting_nonNullOriginCat["gold_label_encoded"] == majority_voting_nonNullOriginCat["majority_label_encoded"]

print("t-test for gold match percentage (general, majority vote)")
all_gold_match_perc = np.asarray([float(num) for num in majority_voting_nonNullOriginCat["is_human_match"]])
t_statistic, p_value = stats.ttest_1samp(a=all_gold_match_perc, popmean=0.5) 
print("t-statistic:", t_statistic)
print("p-value:", p_value)

print('Comparing MTURKERS AND OLD GOLD STANDARD LABELS')
performance_metrics(majority_voting_nonNullOriginCat, topicID='OVERALL')
performance_metrics(majority_voting_nonNullOriginCat, topicID='min_wage')
performance_metrics(majority_voting_nonNullOriginCat, topicID='gun_control')

print('PRO VIDEOS: Comparing MTURKERS AND OLD GOLD STANDARD LABELS')
performance_metrics(pro_only_maj_vote, topicID='OVERALL')
performance_metrics(pro_only_maj_vote, topicID='min_wage')
performance_metrics(pro_only_maj_vote, topicID='gun_control')

print('ANTI VIDEOS: Comparing MTURKERS AND OLD GOLD STANDARD LABELS')
performance_metrics(anti_only_maj_vote, topicID='OVERALL')
performance_metrics(anti_only_maj_vote, topicID='min_wage')
performance_metrics(anti_only_maj_vote, topicID='gun_control')


# # Gold Standard v. Majority Label (Dropping Insufficient Data)

def performance_metrics_drop_insuf(filtered_df, topicID='OVERALL'):
    print('**********')
    print(f'{topicID}')

    if topicID == 'OVERALL':
        accuracy = accuracy_score(filtered_df['gold_label_encoded'], 
                                  filtered_df['majority_label_drop_insuf_encoded'])
        precision = precision_score(filtered_df['gold_label_encoded'], 
                                    filtered_df['majority_label_drop_insuf_encoded'], 
                                    labels = [0,1],
                                    average='weighted',
                                    zero_division=0)
        recall = recall_score(filtered_df['gold_label_encoded'], 
                              filtered_df['majority_label_drop_insuf_encoded'],
                              labels = [0,1], 
                              average='weighted',
                              zero_division=0)
    else:
        accuracy = accuracy_score(filtered_df[filtered_df.topicID == f'{topicID}']['gold_label_encoded'], 
                                  filtered_df[filtered_df.topicID == f'{topicID}']['majority_label_drop_insuf_encoded'])
        precision = precision_score(filtered_df[filtered_df.topicID == f'{topicID}']['gold_label_encoded'], 
                                    filtered_df[filtered_df.topicID == f'{topicID}']['majority_label_drop_insuf_encoded'],
                                    labels = [0,1], 
                                    average='weighted',
                                    zero_division=0)
        recall = recall_score(y_true = filtered_df[filtered_df.topicID == f'{topicID}']['gold_label_encoded'], 
                              y_pred = filtered_df[filtered_df.topicID == f'{topicID}']['majority_label_drop_insuf_encoded'], 
                              labels = [0,1],
                              average= 'weighted',
                              zero_division=0)
                
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print('**********')

print('Comparing MTURKERS AND OLD GOLD STANDARD LABELS')
performance_metrics_drop_insuf(majority_voting_nonNullOriginCat, topicID='OVERALL')
performance_metrics_drop_insuf(majority_voting_nonNullOriginCat, topicID='min_wage')
performance_metrics_drop_insuf(majority_voting_nonNullOriginCat, topicID='gun_control')

print('PRO VIDEOS: Comparing MTURKERS AND OLD GOLD STANDARD LABELS')
performance_metrics_drop_insuf(pro_only_maj_vote, topicID='OVERALL')
performance_metrics_drop_insuf(pro_only_maj_vote, topicID='min_wage')
performance_metrics_drop_insuf(pro_only_maj_vote, topicID='gun_control')

print('ANTI VIDEOS: Comparing MTURKERS AND OLD GOLD STANDARD LABELS')
performance_metrics_drop_insuf(anti_only_maj_vote, topicID='OVERALL')
performance_metrics_drop_insuf(anti_only_maj_vote, topicID='min_wage')
performance_metrics_drop_insuf(anti_only_maj_vote, topicID='gun_control')


# # Gold Standard v. Majority Label ("Easy" Subset)
majority_voting_easyOnly = majority_voting_nonNullOriginCat[majority_voting_nonNullOriginCat["originId"].isin(MINWAGE_AGREED_VIDEOS["originID"])]

# create the pro- and anti-only sets
pro_only_maj_vote_easy = majority_voting_easyOnly[majority_voting_easyOnly['originCat'] == 'pro']
anti_only_maj_vote_easy = majority_voting_easyOnly[majority_voting_easyOnly['originCat'] == 'anti']
# flip all the labels for anti_only
anti_only_maj_vote_easy = anti_only_maj_vote_easy.copy()
anti_only_maj_vote_easy.loc[:, 'majority_label_encoded'] = anti_only_maj_vote_easy['majority_label_encoded'].replace({0: 1, 1: 0})
anti_only_maj_vote_easy.loc[:, 'gold_label_encoded'] = anti_only_maj_vote_easy['gold_label_encoded'].replace({0: 1, 1: 0})
anti_only_maj_vote_easy.loc[:, 'majority_label_drop_insuf_encoded'] = anti_only_maj_vote_easy['majority_label_drop_insuf_encoded'].replace({0: 1, 1: 0})

print('Comparing MTURKERS AND OLD GOLD STANDARD LABELS')
performance_metrics(majority_voting_easyOnly, topicID='OVERALL')

print('PRO VIDEOS: Comparing MTURKERS AND OLD GOLD STANDARD LABELS')
performance_metrics_drop_insuf(pro_only_maj_vote_easy, topicID='OVERALL')

print('ANTI VIDEOS: Comparing MTURKERS AND OLD GOLD STANDARD LABELS')
performance_metrics_drop_insuf(anti_only_maj_vote_easy, topicID='OVERALL')


# # Gold Standard versus GPT
def performance_gpt(filtered_df, topicID='OVERALL'):
    print('**********')
    print(f'{topicID}')

    if topicID == 'OVERALL':
        accuracy = accuracy_score(filtered_df['gold_label_encoded'], 
                                  filtered_df['gpt_encoded'])
        precision = precision_score(filtered_df['gold_label_encoded'], 
                                    filtered_df['gpt_encoded'], 
                                    labels = [0,1,2], 
                                    average='weighted',
                                    zero_division=0)
        recall = recall_score(filtered_df['gold_label_encoded'], 
                              filtered_df['gpt_encoded'], 
                              labels = [0,1,2], 
                              average='weighted',
                              zero_division=0)
    else:
        accuracy = accuracy_score(filtered_df[filtered_df.topicID == f'{topicID}']['gold_label_encoded'], 
                                  filtered_df[filtered_df.topicID == f'{topicID}']['gpt_encoded'])
        precision = precision_score(filtered_df[filtered_df.topicID == f'{topicID}']['gold_label_encoded'], 
                                    filtered_df[filtered_df.topicID == f'{topicID}']['gpt_encoded'], 
                                    labels = [0,1,2], 
                                    average='weighted',
                                    zero_division=0)
        recall = recall_score(filtered_df[filtered_df.topicID == f'{topicID}']['gold_label_encoded'], 
                              filtered_df[filtered_df.topicID == f'{topicID}']['gpt_encoded'], 
                              labels = [0,1,2], 
                              average='weighted',
                              zero_division=0)
        
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print('**********')

majority_voting_nonNullGPT = majority_voting_nonNullOriginCat[majority_voting_nonNullOriginCat.gpt_thumb_rating.isnull()==False]
pro_only_maj_vote = majority_voting_nonNullGPT[majority_voting_nonNullGPT['originCat'] == 'pro']
anti_only_maj_vote = majority_voting_nonNullGPT[majority_voting_nonNullGPT['originCat'] == 'anti']

# flip all the labels for anti_only
anti_only_maj_vote = anti_only_maj_vote.copy()
anti_only_maj_vote.loc[:, 'gold_label_encoded'] = anti_only_maj_vote['gold_label_encoded'].replace({0: 1, 1: 0})
anti_only_maj_vote.loc[:, 'gpt_encoded'] = anti_only_maj_vote['gpt_encoded'].replace({0: 1, 1: 0})

print("number of videos that gpt rated")
print(len(majority_voting_nonNullGPT))

print('Comparing GPT AND OLD GOLD STANDARD LABELS')
performance_gpt(majority_voting_nonNullGPT, topicID='OVERALL')
performance_gpt(majority_voting_nonNullGPT, topicID='min_wage')
performance_gpt(majority_voting_nonNullGPT, topicID='gun_control')

print('PRO VIDEOS: Comparing GPT AND OLD GOLD STANDARD LABELS')
performance_gpt(pro_only_maj_vote, topicID='OVERALL')
performance_gpt(pro_only_maj_vote, topicID='min_wage')
performance_gpt(pro_only_maj_vote, topicID='gun_control')

print('ANTI VIDEOS: Comparing GPT AND OLD GOLD STANDARD LABELS')
performance_gpt(anti_only_maj_vote, topicID='OVERALL')
performance_gpt(anti_only_maj_vote, topicID='min_wage')
performance_gpt(anti_only_maj_vote, topicID='gun_control')

print("t-test for gold match percentage (general, majority vote)")
majority_voting_nonNullGPT.loc[:, "is_human_match"] = majority_voting_nonNullGPT["gold_label_encoded"] == majority_voting_nonNullGPT["majority_label_encoded"]
majority_voting_nonNullGPT.loc[:, "is_human_match_drop_insuf"] = majority_voting_nonNullGPT["gold_label_encoded"] == majority_voting_nonNullGPT["majority_label_drop_insuf_encoded"]
majority_voting_nonNullGPT.loc[:, "is_gpt_match"] = majority_voting_nonNullGPT["gold_label_encoded"] == majority_voting_nonNullGPT["gpt_encoded"]

print("Humans v. GPT")
all_gold_match_perc = np.asarray([float(num) for num in majority_voting_nonNullGPT["is_human_match"]])
t_statistic, p_value = stats.ttest_1samp(a=all_gold_match_perc, popmean=np.mean(majority_voting_nonNullGPT["is_gpt_match"])) 
print("t-statistic:", t_statistic)
print("p-value:", p_value)

print("Humans (with 'Insufficient Data' Dropped) v. GPT")
all_gold_match_perc = np.asarray([float(num) for num in majority_voting_nonNullGPT["is_human_match_drop_insuf"]])
t_statistic, p_value = stats.ttest_1samp(a=all_gold_match_perc, popmean=np.mean(majority_voting_nonNullGPT["is_gpt_match"])) 
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# # What percentage of video thumbnails were 'clearly partisan?'
majority_voting['Majority Vote Percentage'] = majority_voting['num_votes_for_majority'] / majority_voting['vote_count']

# plot a histogram of the majority vote percentage
sns.kdeplot(data=majority_voting, x='Majority Vote Percentage', hue='originCat')
# add a vertical line at 0.5
plt.axvline(0.5, color='r', linestyle='--')
plt.axvline(0.8, color='lightpink', linestyle='--')
plt.show()

majority_voting_wage = majority_voting[majority_voting.topicID == 'min_wage']
majority_voting_gun = majority_voting[majority_voting.topicID == 'gun_control']

sns.kdeplot(data=majority_voting_gun, x='Majority Vote Percentage', hue='originCat')
# add a vertical line at 0.5
plt.axvline(0.5, color='r', linestyle='--')
plt.axvline(0.8, color='lightpink', linestyle='--')
plt.show()


# ## Can we connect it with the original choices people made?

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

tree_files_wage = glob.glob(os.path.join('../recommendation_trees/trees_wage/', '*.csv'))
tree_files_gun = glob.glob(os.path.join('../recommendation_trees/trees_gun/', '*.csv'))


all_trees_by_ID_wage, all_trees_by_channelID_wage = read_all_trees(tree_files_wage)

all_trees_by_ID_gun, all_trees_by_channelID_gun = read_all_trees(tree_files_gun)

# get the thing in parentheses as the topicid
gun_topicids = [re.search(r'\((.*?)\)', filename).group(1) for filename in tree_files_gun]
wage_topicids = [re.search(r'\((.*?)\)', filename).group(1) for filename in tree_files_wage]


# # Thumbnail Distribution

print('Total number of videos shown:', 
      len(videos['gun_control']) + len(videos['min_wage']))

print('Percentage of videos shown: %', 
      np.round((len(videos['gun_control']) + len(videos['min_wage'])) / len(labels_on_platform), 3) * 100)

unique_videos_shown = set(list(videos['gun_control'].keys()) + list(videos['min_wage'].keys()))
unique_videos_in_platform_set = set(labels_on_platform['originId'])

len(labels_on_platform)
len(unique_videos_in_platform_set)
unique_videos_in_platform_set.difference(unique_videos_shown)

video_lengths = {}
for vids, labellist in videos.items():
    for vid, labels in labellist.items():
        video_lengths[vid] = len(labels)


average_number_of_ratings = np.mean([int(val) for val in video_lengths.values()])
average_number_of_ratings

# plot a histogram of video_lengths.values()
plt.hist(video_lengths.values(), bins=40)
# vertical line around the mean (average_number_of_ratings)
plt.axvline(average_number_of_ratings, color='r', linestyle='dashed', linewidth=1)
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Videos')


# # Exploratory Analyses

# ## What happens when we drop the cases where someone (either human or GPT) said there wasn't enough information?
# When a rater abstains by saying there wasn't enough information, it deflates all the metrics, since the gold standard ratings are all binary.

majority_voting_rated = majority_voting_nonNullGPT[(majority_voting_nonNullGPT.majority_label != "insufficient data.") & (majority_voting_nonNullGPT.gpt_thumb_rating != "insufficient data.")]
len(majority_voting_rated) / len(majority_voting_nonNullGPT) # we now have 90% of the original videos

len(majority_voting_rated)

print('Comparing MTURKERS AND OLD GOLD STANDARD LABELS -- FILTERED')
performance_metrics(majority_voting_rated, topicID='OVERALL')
performance_metrics(majority_voting_rated, topicID='min_wage')
performance_metrics(majority_voting_rated, topicID='gun_control')

print('Comparing GPT AND OLD GOLD STANDARD LABELS -- FILTERED')
performance_gpt(majority_voting_rated, topicID='OVERALL')
performance_gpt(majority_voting_rated, topicID='min_wage')
performance_gpt(majority_voting_rated, topicID='gun_control')


# ## Are there any weird patterns that we should filter out?

# For example, we might want to explore cases where people answered the same thing all 20 times, or always said 'insufficient information.'

INDEXES_TO_EXCLUDE = set() # keep track of indices that we should exclude for various reasons

exp_indexes_wage, exp_indexes_gun = session_rep_counts(real_data)
indexes = exp_indexes_wage + exp_indexes_gun 

for index in indexes:
        
    ratings = real_data[index]['ratingResults']
    ratings_indices = [ratings[i]['index'] for i in range(0, len(ratings))]

    # These people rated the same thing for all questions
    if len(set(ratings_indices)) == 1:
        INDEXES_TO_EXCLUDE.add(index)

    # These people saw the same video multiple times but had inconsistent answers
    rating_dict_for_individual = {}
    for rating in ratings:
        if(rating['vid'] not in rating_dict_for_individual.keys()):
            rating_dict_for_individual[rating['vid']] = []
        rating_dict_for_individual[rating['vid']].append(rating['index'])

    # identify if any keys in rating_dict_for_individual have a length greater than 1
    for key in rating_dict_for_individual.keys():
        if len(rating_dict_for_individual[key]) > 1:
            if(len(set(rating_dict_for_individual[key])) > 1): # these people had inconsistent responses when rating the same video
                INDEXES_TO_EXCLUDE.add(index)


# indices that we exclude for the above data quality reasons (button-smashing and inconsistent responses)
INDEXES_TO_EXCLUDE

indexes_cleaned = [index for index in indexes if index not in INDEXES_TO_EXCLUDE]

# filter real_data to indexes_cleaned
real_data_cleaned = [real_data[index] for index in indexes_cleaned]
result_df_cleaned = thumbnail_exp_check(real_data_cleaned)

# We don't actually do much better because there were only 7 participants dropped for data quality issues
result_df_cleaned.groupby('topic_id').agg(
    session_count=('session_id', 'nunique'),
    respondent_count=('respondent_id', 'nunique'),
    gold_match_mean=('gold_match_perc', 'mean'),
    gold_match_std=('gold_match_perc', 'std'),
    gold_match_quartiles=('gold_match_perc', calculate_quartiles)
).reset_index()
