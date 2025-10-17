#!/usr/bin/env python
# coding: utf-8

# # Expected Value of Partisanship Increase

# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import t

print('=' * 80 + '\n\n' + 'OUTPUT FROM: supplemental/increasingly extreme recommendations/10_partisanship_increase.py' + '\n\n')


# Data from Youtube Rec
wage_data = pd.read_csv('../data/supplemental/metadata and ratings/metadata_with_lables_binary_only_checked_0410.csv')

# GPT label (continuous) --> full tree data
gpt_labels_cont = pd.read_csv("../data/supplemental/metadata and ratings/gpt_continuous_ratings_minwage_FULL_averaged (1).csv")[['naijia_originId','gpt_continuous']]
gpt_labels_cont = gpt_labels_cont.rename(columns={'naijia_originId':'originID','gpt_continuous':'gpt_label'})

# eliminate duplicates if any
gpt_labels_cont = gpt_labels_cont.groupby('originID').agg({"gpt_label":"mean"}).reset_index()

# ## Min Wage Increase - average recommendation on the average video

pairs = wage_data[['originID','recID','step']]
pairs = pairs.merge(gpt_labels_cont, how='left',on='originID').rename(columns={'gpt_label':'gpt_label_originID'})


# data format: Each row is a (current, recommended) video pair
# cols: cur video ID, cur video rating, cur video rec, cur video rec rating
pairs = pairs.merge(gpt_labels_cont, 
                    how='left',
                    left_on='recID', 
                    right_on='originID').rename(columns={'gpt_label':'gpt_label_recID',
                                                         'originID_x':'originID'}).drop(columns=['originID_y'])

pairs = pairs[(pairs.gpt_label_originID.isnull() == False) & 
(pairs.gpt_label_recID.isnull() == False)]

# weight videos by the number of recommendations they have
weights = pairs.groupby('originID').agg({"recID":
                                         "nunique"}).reset_index().rename(columns={"recID":
                                                                                   "weight"})
pairs = pairs.merge(weights, how='left',on='originID')

# Difference = Recommended Score − Current Score
pairs['difference'] = pairs['gpt_label_recID'] - pairs['gpt_label_originID']
pairs['weighted_difference'] = pairs['difference'] / pairs['weight']

## liberal/conservative categorization
def label_category(row):
    if row['gpt_label_originID'] > 0:
        return 'conservative'
    else:
        return 'liberal'

pairs['label_category'] = pairs.apply(label_category, axis=1)

# Liberal Cur Videos
liberal = pairs[pairs.label_category == 'liberal']


# Constant for the intercept
lib_X = sm.add_constant(liberal['gpt_label_originID'])

# OLS model with two-way clustering
model = sm.OLS(liberal['weighted_difference'], lib_X)
lib_results = model.fit(cov_type='cluster', 
                    cov_kwds={'groups': [liberal['originID'].tolist(), liberal['recID'].tolist()]})
print(lib_results.summary())


# Conservative Cur Videos
conservative = pairs[pairs.label_category == 'conservative']

# Constant for the intercept
cons_X = sm.add_constant(conservative['gpt_label_originID'])

# OLS model with two-way clustering
model = sm.OLS(conservative['weighted_difference'], cons_X)
cons_results = model.fit(cov_type='cluster', 
                    cov_kwds={'groups': [conservative['originID'].tolist(), conservative['recID'].tolist()]})
print(cons_results.summary())


### SI Figure

fig, ax = plt.subplots()

lib_preds = lib_results.get_prediction(lib_X).summary_frame(alpha=0.05)
cons_preds = cons_results.get_prediction(cons_X).summary_frame(alpha=0.05)

ax.scatter(liberal['gpt_label_originID'], 
            liberal['gpt_label_originID'] + liberal['weighted_difference'], 
            color='lightblue', s=0.9, alpha=0.6)

ax.scatter(conservative['gpt_label_originID'], 
            conservative['gpt_label_originID'] + conservative['weighted_difference'], 
            color='lightcoral', s=0.9, alpha=0.6)

ax.fill_between(liberal['gpt_label_originID'], 
                 liberal['gpt_label_originID'] + lib_preds['mean_ci_lower'], 
                 liberal['gpt_label_originID'] + lib_preds['mean_ci_upper'], 
                alpha=.4, color='blue')

ax.fill_between(conservative['gpt_label_originID'], 
                 conservative['gpt_label_originID'] + cons_preds['mean_ci_lower'], 
                 conservative['gpt_label_originID'] + cons_preds['mean_ci_upper'], 
                alpha=.4, color='red')

ax.plot(liberal['gpt_label_originID'], 
         lib_preds['mean'] + liberal['gpt_label_originID'], 
         color='darkblue', linewidth=0.5)

ax.plot(conservative['gpt_label_originID'],
         cons_preds['mean'] + conservative['gpt_label_originID'], 
         color='darkred', linewidth=0.5)

ax.plot([-1, 1], [-1, 1], 'k--', linewidth=1.5)

# Customize the plot
ax.set_xlabel('Current Video Rating')
ax.set_ylabel('Recommended Video Rating')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.grid(False)
ax.set_title('Current Video Rating vs \n Recommended Video Rating')
plt.savefig('../results/video_rating_pairs.png', dpi=300, bbox_inches='tight')



