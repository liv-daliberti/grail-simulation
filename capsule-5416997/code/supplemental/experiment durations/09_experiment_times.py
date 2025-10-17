### Average Time Spent in Experiments

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from IPython.display import Image

print('=' * 80 + '\n\n' + 'OUTPUT FROM: supplemental/experiment durations/09_experiment_times.py' + '\n\n')

study1 = pd.read_csv('../results/intermediate data/gun control (issue 1)/guncontrol_qualtrics_w123_clean.csv')
study2 = pd.read_csv('../results/intermediate data/minimum wage (issue 2)/qualtrics_w12_clean.csv')
study3 = pd.read_csv('../results/intermediate data/minimum wage (issue 2)/yg_w12_clean.csv')
study4 = pd.read_csv('../results/intermediate data/shorts/qualtrics_w12_clean_ytrecs_may2024.csv')

# ## Outlier Elimination
# Manually patching the guaranteed incorrect values 
# by taking the interface end time to be `pmin(survey end time, raw interface end time)`
# Create a new column 'interface_end_time' by taking the minimum of the two columns

# Convert 'end_date_w2' and 'end_time2' to datetime format
study1['end_date_w2'] = pd.to_datetime(study1['end_date_w2'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
study1['end_time2'] = pd.to_datetime(study1['end_time2'], format='%Y-%m-%dT%H:%M:%SZ',errors='coerce').dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
study1['start_time2'] = pd.to_datetime(study1['start_time2'], format='%Y-%m-%dT%H:%M:%SZ',errors='coerce').dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)

# Create a new column 'interface_end_time_fixed' by taking the minimum of both dates
study1['interface_end_time_fixed'] = study1['end_date_w2'].combine(study1['end_time2'], 
    lambda x, y: x if pd.notna(y) and (pd.isna(x) or x < y) else y)

study1['interface_end_time_fixed'] = study1['interface_end_time_fixed'].where(
    pd.notna(study1['interface_end_time_fixed']), np.nan
)

# Convert 'end_date_w2' and 'end_time2' to datetime format
study2['end_date_w2'] = pd.to_datetime(study2['end_date_w2'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
study2['end_time2'] = pd.to_datetime(study2['end_time2'], format='%Y-%m-%dT%H:%M:%SZ',errors='coerce').dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
study2['start_time2'] = pd.to_datetime(study2['start_time2'], format='%Y-%m-%dT%H:%M:%SZ',errors='coerce').dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)

# Create a new column 'interface_end_time_fixed' by taking the minimum of both dates
study2['interface_end_time_fixed'] = study2['end_date_w2'].combine(study2['end_time2'], 
    lambda x, y: x if pd.notna(y) and (pd.isna(x) or x < y) else y)

study2['interface_end_time_fixed'] = study2['interface_end_time_fixed'].where(
    pd.notna(study2['interface_end_time_fixed']), np.nan
)

# Convert 'end_date_w2' and 'end_time2' to datetime format
study3['start_date_w2'] = pd.to_datetime(study3['start_date_w2'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize(None)
study3['end_date_w2'] = pd.to_datetime(study3['end_date_w2'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize(None)
study3['end_time2'] = pd.to_datetime(study3['end_time2'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize(None)
study3['start_time2'] = pd.to_datetime(study3['start_time2'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize(None)
# Create a new column 'interface_time_fixed' with the minimum of both dates
study3['interface_end_time_fixed'] = np.minimum(study3['end_date_w2'], study3['end_time2'])

# Fixed duration values for study 4 is done.
study4['interface_time_fixed'] = np.minimum(study4['survey_time'], study4['interface_duration']/60)

# Ensure both columns are in datetime format, and make them timezone-naive
study1['interface_end_time_fixed'] = pd.to_datetime(study1['interface_end_time_fixed'], errors='coerce').dt.tz_localize(None)
study1['start_time2'] = pd.to_datetime(study1['start_time2'], errors='coerce').dt.tz_localize(None)
# Subtract the times while handling nulls
study1['interface_time_fixed'] = study1['interface_end_time_fixed'] - study1['start_time2']

# Ensure both columns are in datetime format, and make them timezone-naive
study2['interface_end_time_fixed'] = pd.to_datetime(study2['interface_end_time_fixed'], errors='coerce').dt.tz_localize(None)
study2['start_time2'] = pd.to_datetime(study2['start_time2'], errors='coerce').dt.tz_localize(None)
# Subtract the times while handling nulls
study2['interface_time_fixed'] = study2['interface_end_time_fixed'] - study2['start_time2']

# Ensure both columns are in datetime format, and make them timezone-naive
study3['interface_end_time_fixed'] = pd.to_datetime(study3['interface_end_time_fixed'], errors='coerce').dt.tz_localize(None)
study3['start_time2'] = pd.to_datetime(study3['start_time2'], errors='coerce').dt.tz_localize(None)
# Subtract the times while handling nulls
study3['interface_time_fixed'] = study3['interface_end_time_fixed'] - study3['start_time2']

# Convert the timedelta to minutes
study1['interface_time_fixed_minutes'] = study1['interface_time_fixed'].dt.total_seconds() / 60
study2['interface_time_fixed_minutes'] = study2['interface_time_fixed'].dt.total_seconds() / 60
study3['interface_time_fixed_minutes'] = study3['interface_time_fixed'].dt.total_seconds() / 60


#### Windsorization

# Copy the 'duration' column to a new 'platform_duration' column
study1['platform_duration'] = study1['duration']
study2['platform_duration'] = study2['duration']
study3['platform_duration'] = study3['duration']
study4['platform_duration'] = study4['interface_duration']

# Calculate the 2.5% and 97.5% quantiles
lower_quantile_study1 = study1['duration'].quantile(0.025)
upper_quantile_study1 = study1['duration'].quantile(0.975)

lower_quantile_study2 = study2['duration'].quantile(0.025)
upper_quantile_study2 = study2['duration'].quantile(0.975)

lower_quantile_study3 = study3['duration'].quantile(0.025)
upper_quantile_study3 = study3['duration'].quantile(0.975)

lower_quantile_study4 = study4['interface_duration'].quantile(0.025)
upper_quantile_study4 = study4['interface_duration'].quantile(0.975)

# Apply Windsorization: cap the values at 2.5% and 97.5%
study1['platform_duration'] = study1['platform_duration'].apply(
    lambda x: lower_quantile_study1 if x <= lower_quantile_study1 else upper_quantile_study1 if x >= upper_quantile_study1 else x
)

study2['platform_duration'] = study2['platform_duration'].apply(
    lambda x: lower_quantile_study2 if x <= lower_quantile_study2 else upper_quantile_study2 if x >= upper_quantile_study2 else x
)

study3['platform_duration'] = study3['platform_duration'].apply(
    lambda x: lower_quantile_study3 if x <= lower_quantile_study3 else upper_quantile_study3 if x >= upper_quantile_study3 else x
)

study4['platform_duration'] = study4['platform_duration'].apply(
    lambda x: lower_quantile_study4 if x <= lower_quantile_study4 else upper_quantile_study4 if x >= upper_quantile_study4 else x
)

# Copy the 'duration' column to a new 'platform_duration' column
study1['platform_duration'] = study1['interface_time_fixed_minutes']
study2['platform_duration'] = study2['interface_time_fixed_minutes']
study3['platform_duration'] = study3['interface_time_fixed_minutes']
study4['platform_duration'] = study4['interface_time_fixed']

# Calculate the 2.5% and 97.5% quantiles
lower_quantile_study1 = study1['interface_time_fixed_minutes'].quantile(0.025)
upper_quantile_study1 = study1['interface_time_fixed_minutes'].quantile(0.975)

lower_quantile_study2 = study2['interface_time_fixed_minutes'].quantile(0.025)
upper_quantile_study2 = study2['interface_time_fixed_minutes'].quantile(0.975)

lower_quantile_study3 = study3['interface_time_fixed_minutes'].quantile(0.025)
upper_quantile_study3 = study3['interface_time_fixed_minutes'].quantile(0.975)

lower_quantile_study4 = study4['interface_time_fixed'].quantile(0.025)
upper_quantile_study4 = study4['interface_time_fixed'].quantile(0.975)

# Apply Windsorization: cap the values at 2.5% and 97.5%
study1['platform_duration'] = study1['platform_duration'].apply(
    lambda x: lower_quantile_study1 if x <= lower_quantile_study1 else upper_quantile_study1 if x >= upper_quantile_study1 else x
)

study2['platform_duration'] = study2['platform_duration'].apply(
    lambda x: lower_quantile_study2 if x <= lower_quantile_study2 else upper_quantile_study2 if x >= upper_quantile_study2 else x
)

study3['platform_duration'] = study3['platform_duration'].apply(
    lambda x: lower_quantile_study3 if x <= lower_quantile_study3 else upper_quantile_study3 if x >= upper_quantile_study3 else x
)

study4['platform_duration'] = study4['platform_duration'].apply(
    lambda x: lower_quantile_study4 if x <= lower_quantile_study4 else upper_quantile_study4 if x >= upper_quantile_study4 else x
)

# Overall interface time spent (mean, Studies 1-3)
print('Mean Interface Time for Studies 1-3:', pd.concat([study1[study1.treatment_arm != 'control']['platform_duration'], 
                                                      study2[study2.treatment_arm != 'control']['platform_duration'], 
                                                      study3[study3.treatment_arm != 'control']['platform_duration']], 
                                                     ignore_index=True).mean())

print('******')
# Overall interface time spent (mean, Studies 1-4)
print('Mean Interface Time for Studies 1-4:', pd.concat([study1[study1.treatment_arm != 'control']['platform_duration'], 
                                                      study2[study2.treatment_arm != 'control']['platform_duration'], 
                                                      study3[study3.treatment_arm != 'control']['platform_duration'],
                                                      study4['platform_duration']], 
                                                     ignore_index=True).mean())

print('******')
# Interface time spent each
print('Study1 Interface:',study1[study1.treatment_arm != 'control']['platform_duration'].mean())
print('Study2 Interface:',study2[study2.treatment_arm != 'control']['platform_duration'].mean())
print('Study3 Interface:',study3[study3.treatment_arm != 'control']['platform_duration'].mean())
print('Study4 Interface:',study4['platform_duration'].mean())


### Plots

# Enable the pandas-to-R conversion
pandas2ri.activate()

# Convert the DataFrame to R DataFrame
w123_r = pandas2ri.py2rpy(study1)

# Define the R code to create the plot and save it as an image
r_code = """
library(ggplot2)
library(dplyr)

# Filter the data
w123_filtered <- w123 %>% filter(treatment_arm != "control")

# Create the plot and save it as a PNG file
surveytime_plot <- ggplot(w123_filtered) +
    geom_histogram(aes(x = platform_duration, y = ..density.. / sum(..density..))) +
    scale_x_continuous("Interface Time Taken (minutes),\nexcluding control respondents", breaks = seq(0, 100, 20), limits = c(-1, 101)) +
    scale_y_continuous("Density") +
    geom_vline(xintercept = mean(w123_filtered$platform_duration, na.rm = TRUE), linetype = "dashed", color = "red") +
    annotate("text", x = mean(w123_filtered$platform_duration, na.rm = TRUE) + 1, y = 0.13, label = paste0("Average: ", round(mean(w123_filtered$platform_duration, na.rm = TRUE), 0), " minutes"), hjust = 0) +
    geom_vline(xintercept = median(w123_filtered$platform_duration, na.rm = TRUE), linetype = "dotted", color = "red") +
    annotate("text", x = median(w123_filtered$platform_duration , na.rm = TRUE) + 1, y = 0.16, label = paste0("Median: ", round(median(w123_filtered$platform_duration, na.rm = TRUE), 0), " minutes"), hjust = 0) +
    theme_minimal()
ggsave(surveytime_plot,filename = "../results/video_platform_duration_study1.pdf",height=3,width=5)
"""

# Load the DataFrame into the R environment & run
robjects.globalenv['w123'] = w123_r
robjects.r(r_code)
#Image(filename="video_platform_duration_study1.png")

# Enable the pandas-to-R conversion
pandas2ri.activate()

# Convert the dataframe to R DataFrame
w123_r = pandas2ri.py2rpy(study2)

# Define the R code to create the plot and save it as an image
r_code = """
library(ggplot2)
library(dplyr)

# Filter the data
w123_filtered <- w123 %>% filter(treatment_arm != "control")

# Create the plot and save it as a PNG file
surveytime_plot <- ggplot(w123_filtered) +
    geom_histogram(aes(x = platform_duration, y = ..density.. / sum(..density..))) +
    scale_x_continuous("Interface Time Taken (minutes),\nexcluding control respondents", breaks = seq(0, 100, 20), limits = c(-1, 101)) +
    scale_y_continuous("Density") +
    geom_vline(xintercept = mean(w123_filtered$platform_duration, na.rm = TRUE), linetype = "dashed", color = "red") +
    annotate("text", x = mean(w123_filtered$platform_duration, na.rm = TRUE) + 1, y = 0.13, label = paste0("Average: ", round(mean(w123_filtered$platform_duration, na.rm = TRUE), 0), " minutes"), hjust = 0) +
    geom_vline(xintercept = median(w123_filtered$platform_duration, na.rm = TRUE), linetype = "dotted", color = "red") +
    annotate("text", x = median(w123_filtered$platform_duration , na.rm = TRUE) + 1, y = 0.16, label = paste0("Median: ", round(median(w123_filtered$platform_duration, na.rm = TRUE), 0), " minutes"), hjust = 0) +
    theme_minimal()

ggsave(surveytime_plot,filename = "../results/video_platform_duration_study2.pdf",height=3,width=5)
"""

robjects.globalenv['w123'] = w123_r
robjects.r(r_code)
#Image(filename="video_platform_duration_study2.png")

# Enable the pandas-to-R conversion
pandas2ri.activate()
w123_r = pandas2ri.py2rpy(study3)

# Define the R code to create the plot and save it as an image
r_code = """
library(ggplot2)
library(dplyr)

# Filter the data
w123_filtered <- w123 %>% filter(treatment_arm != "control")

# Create the plot and save it as a PNG file
surveytime_plot <- ggplot(w123_filtered) +
    geom_histogram(aes(x = platform_duration, y = ..density.. / sum(..density..))) +
    scale_x_continuous("Interface Time Taken (minutes),\nexcluding control respondents", breaks = seq(0, 100, 20), limits = c(-1, 101)) +
    scale_y_continuous("Density") +
    geom_vline(xintercept = mean(w123_filtered$platform_duration, na.rm = TRUE), linetype = "dashed", color = "red") +
    annotate("text", x = mean(w123_filtered$platform_duration, na.rm = TRUE) + 1, y = 0.13, label = paste0("Average: ", round(mean(w123_filtered$platform_duration, na.rm = TRUE), 0), " minutes"), hjust = 0) +
    geom_vline(xintercept = median(w123_filtered$platform_duration, na.rm = TRUE), linetype = "dotted", color = "red") +
    annotate("text", x = median(w123_filtered$platform_duration , na.rm = TRUE) + 1, y = 0.16, label = paste0("Median: ", round(median(w123_filtered$platform_duration, na.rm = TRUE), 0), " minutes"), hjust = 0) +
    theme_minimal()
ggsave(surveytime_plot,filename = "../results/video_platform_duration_study3.pdf",height=3,width=5)
"""

# Load the DataFrame into the R environment
robjects.globalenv['w123'] = w123_r
robjects.r(r_code)
#Image(filename="video_platform_duration_study3.png")


pandas2ri.activate()
w123_r = pandas2ri.py2rpy(study4)

# Define the R code to create the plot and save it as an image
r_code = """
library(ggplot2)
library(dplyr)

# Filter the data
w123_filtered <- w123 %>% filter(treatment_arm != "control")

# Create the plot and save it as a PNG file
surveytime_plot <- ggplot(w123_filtered) +
    geom_histogram(aes(x = platform_duration, y = ..density.. / sum(..density..))) +
    scale_x_continuous("Interface Time Taken (minutes)", breaks = seq(0, 100, 20), limits = c(-1, 101)) +
    scale_y_continuous("Density") +
    geom_vline(xintercept = mean(w123_filtered$platform_duration, na.rm = TRUE), linetype = "dashed", color = "red") +
    annotate("text", x = mean(w123_filtered$platform_duration, na.rm = TRUE) + 1, y = 0.13, label = paste0("Average: ", round(mean(w123_filtered$platform_duration, na.rm = TRUE), 0), " minutes"), hjust = 0) +
    geom_vline(xintercept = median(w123_filtered$platform_duration, na.rm = TRUE), linetype = "dotted", color = "red") +
    annotate("text", x = median(w123_filtered$platform_duration , na.rm = TRUE) + 1, y = 0.16, label = paste0("Median: ", round(median(w123_filtered$platform_duration, na.rm = TRUE), 0), " minutes"), hjust = 0) +
    theme_minimal()
ggsave(surveytime_plot,filename = "../results/video_platform_duration_study4.pdf",height=3,width=5)
"""

robjects.globalenv['w123'] = w123_r
robjects.r(r_code)
#Image(filename="video_platform_duration_study4.png")




