#!/usr/bin/env python
# coding: utf-8

"""
This file serves to plot Figures S12 and S13, which show the robustness of the GPT-generated ratings to different
ways of quantifying a video's political extremeness (BERT ratings and Hosseinmardi's channel labels.)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print('=' * 80 + '\n\n' + 'OUTPUT FROM: supplemental/increasingly extreme recommendations/11_gpt_rating_plots.py' + '\n\n')

# Read in the GPT continuous ratings, along with the other ways of quantifying political expremeness
gpt_rating_wage = pd.read_csv("../data/supplemental/metadata and ratings/gpt_continuous_ratings_minwage.csv")
wage_videos_full = pd.read_csv("../data/supplemental/metadata and ratings/bert_rated_wage_videos_all.csv")

wage_all_merged = wage_videos_full.merge(gpt_rating_wage, on="naijia_originId")
# deduplicate
wage_all_merged = wage_all_merged[["naijia_originId", "homa_explanation_new", "gpt_label", "bert_score", "originCat"]].drop_duplicates()


"""
FIGURE S12: COMPARISON TO BERT MODEL FROM LAI ET AL. (2024)
"""
print("starting Figure S12...")

plt.figure(figsize=(8, 6))
colors = {'pro': 'blue', 'anti': 'red'}

plt.scatter(wage_all_merged['gpt_label'], wage_all_merged['bert_score'], c=wage_all_merged['originCat'].map(colors), alpha=0.7)

plt.axhline(0, color='gray', linestyle='--', linewidth=1.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=1.5)

plt.xlabel('GPT Continuous')
plt.ylabel('BERT Score')
plt.title('Correlation between GPT Continuous and BERT Score')

for cat, color in colors.items():
    plt.scatter([], [], c=color, label=cat)
plt.legend()

plt.grid(True)
plt.savefig('../results/figure_S12_comparison_to_Lai_et_al_bert_ratings.png')

print("...done!")


"""
FIGURE S13: COMPAIRSON WITH HOSSEINMARDI ET AL. (2021)
"""
print("starting Figure S13...")

fig, ax = plt.subplots(figsize=(12, 6))

for category, data in wage_all_merged.groupby('homa_explanation_new'):
    sns.histplot(data=data, x='gpt_label', label=category, kde=True, ax=ax, alpha=0.7)

plt.title("Distribution of GPT Continuous Scores by Hosseinmardi et al. (2021)", fontsize=16)
plt.xlabel('GPT Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(title='Category', fontsize=12, title_fontsize=14)

plt.xlim(-1, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax.grid(True, linestyle='--', alpha=0.5)
sns.despine()
plt.tight_layout()
plt.savefig('../results/figure_S13_comparison_to_Hosseinmardi_et_al_channel_ratings.png')

print("...done!")