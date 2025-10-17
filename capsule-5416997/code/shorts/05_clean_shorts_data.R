cat(rep('=', 80),
    '\n\n',
    'OUTPUT FROM: shorts/05_clean_shorts_data.R',
    '\n\n',
    sep = ''
    )
    
## Extremizing Sequences and Minimum Wage Opinions
## Data collected May 2024 via MTurk/CloudResearch
## Analysis for the Extremizing Sequences Experiment

## Preamble ----------------------------
library(tidyverse)
library(janitor)
library(lubridate)
library(stargazer)
library(broom)

# create a folder for the shorts intermediate data
dir.create("../results/intermediate data/shorts/", recursive = TRUE, showWarnings = FALSE)

# SURVEY DATA (FROM QUALTRICS)
a <- read_csv("../data/shorts/ytrecs_surveys_may2024.csv")[-c(1,2),] %>%
  clean_names() # 1315 obs.

# DATE FILTER
a <- a %>% filter(start_date >= '2024-05-28') # 1032 obs.

# ATTENTION CHECK -- 932 obs.
a <- a %>% filter(a$q81 == "Quick and easy")
a <- a %>% filter(a$q82 == "wikiHow")
a <- a %>% filter(is.na(video_link) == FALSE) ## failed respondents don't have a valid link

# SURVEY TIME (ALL)
a <- a %>% mutate(start_date = as_datetime(start_date),
                  end_date = as_datetime(end_date),
                  survey_time = as.numeric(end_date-start_date))

summary(a$survey_time) # 5.5 mins to 74 mins (median 34 mins)

# DEMOGRAPHICS -------------------------------------------------

# GENDER, EDUCATION, INCOME
a <- a %>%
  mutate(female = ifelse(gender == "Woman", 1, 0),
         male = ifelse(gender == "Man", 1, 0),
         black = ifelse(str_detect(race_ethnicity, "Black"), 1, 0),
         white = ifelse(str_detect(race_ethnicity, "White"), 1, 0),
         college = ifelse(str_detect(highest_education, "college ") | str_detect(highest_education, "Post"), 1, 0),
         income_gt50k = ifelse(income %in% names(table(a$income))[c(2,3,5,10,11,12,13)], 1, 0)
  )
a$income_gt50k[is.na(a$income)] <- NA

# PID
a <- a %>%
  mutate(pid = case_when(pid1=="Democrat" ~ -1,
                         pid1=="Republican" ~ 1,
                         pid4=="Closer to the Republican Party" ~ 1,
                         pid4=="Closer to the Democratic Party" ~ -1,
                         pid4=="Neither" ~ 0))

tabyl(a,pid)

# IDEO
a <- a %>%
  mutate(ideo = case_when(ideo1=="Liberal" ~ -1,
                          ideo1=="Conservative" ~ 1,
                          ideo4=="Closer to conservatives" ~ 1,
                          ideo4=="Closer to liberals" ~ -1,
                          ideo4=="Neither" ~ 0))

tabyl(a,ideo)

# AGE
a$age <- 2024-as.numeric(a$year_born)

# AGE CATEGORIES: 18-29;  30-44;  45-64;  65+
a <- a %>%
  mutate(age_cat = case_when(age>=18 & age<=29 ~ "18-29",
                             age>=30 & age<=44 ~ "30-44",
                             age>=45 & age<=64 ~ "45-64",
                             age>=65 ~ "65+"
  ))
a <- a %>%
  fastDummies::dummy_cols(select_columns = "age_cat",remove_selected_columns = F)

# POLITICAL INTEREST AND YOUTUBE FREQUENCY RECODING
a <- a %>%
  mutate(pol_interest = dplyr::recode(political_interest,"Extremely interested"=5,"Very interested"=4,"Somewhat interested"=3,"Not very interested"=2,"Not at all interested"=1),
         freq_youtube = dplyr::recode(youtube_time,"More than 3 hours per day"=6,"2–3 hours per day"=5,"1–2 hours per day"=4,"31–59 minutes per day"=3,"10–30 minutes per day"=2,"Less than 10 minutes per day"=1,"None"=0)
  )

# SUMMARY TABLE FOR DEMOGRAPHICS
summary_tab <- a %>%
  dplyr::summarize(female = mean(female,na.rm=T),
                   white = mean(white,na.rm=T),
                   black = mean(black,na.rm=T),
                   age1829 = mean(`age_cat_18-29`,na.rm=T),
                   age3044 = mean(`age_cat_30-44`,na.rm=T),
                   age4564 = mean(`age_cat_45-64`,na.rm=T),
                   age65p = mean(`age_cat_65+`,na.rm=T),
                   college = mean(college,na.rm=T),
                   income_gt50k = mean(income_gt50k,na.rm=T),
                   democrat = mean(pid==-1,na.rm=T),
                   republican = mean(pid==1,na.rm=T))

summary_tab <- pivot_longer(summary_tab,
                            cols=c(female,
                                   white,
                                   black,
                                   age1829,
                                   age3044,
                                   age4564,
                                   age65p,
                                   college,
                                   income_gt50k,
                                   democrat,
                                   republican),
                            names_to = "outcome",values_to = "survey_avg")
outcome_labels <- data.frame(outcome_pretty = c("Female",
                                                "White",
                                                "Black",
                                                "Age 18-29",
                                                "Age 30-44",
                                                "Age 45-64",
                                                "Age 65+",
                                                "College educated",
                                                "Income >$50k",
                                                "Democrat",
                                                "Republican"),
                             outcome = c("female",
                                         "white",
                                         "black",
                                         "age1829",
                                         "age3044",
                                         "age4564",
                                         "age65p",
                                         "college",
                                         "income_gt50k",
                                         "democrat",
                                         "republican"))

summary_tab$outcome_pretty <-  outcome_labels$outcome_pretty[match(summary_tab$outcome,outcome_labels$outcome)]
summary_tab <- summary_tab %>%
  mutate(outcome_pretty = factor(outcome_pretty,levels = c("Republican",
                                                           "Democrat",
                                                           "Income >$50k",
                                                           "College educated",
                                                           "Age 65+",
                                                           "Age 45-64",
                                                           "Age 30-44",
                                                           "Age 18-29",
                                                           "Female",
                                                           "Black",
                                                           "White"),ordered=T))

# DEMOGRAPHICS DESCRIPTIVE FIGURE
(descrip_fig <- ggplot(summary_tab) + 
    geom_point(aes(y=outcome_pretty,x=survey_avg)) + 
    geom_text(aes(y=outcome_pretty,x=survey_avg,label=paste0(round(100*survey_avg,0),"%")),nudge_x = 0.1) + 
    scale_y_discrete("") + 
    scale_x_continuous("",labels=scales::percent_format(),limits=c(0,1)) + 
    theme_bw()
)
ggsave(descrip_fig,filename = "../results/shorts_demographics.pdf",height=5,width=4)


### DEMOGRAPHICS DONE ###

#### OUTCOMES ####

##### POLICY OPINIONS #####

# convert to numeric unit scale:
a <- a %>%
  mutate( # higher = more conservative or anti-min wage
    minwage15_pre = dplyr::recode(minwage15_pre,"Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    rtwa_v1_pre = dplyr::recode(rtwa_v1_pre, "Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    rtwa_v2_pre = dplyr::recode(rtwa_v2_pre, "Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    mw_support_pre = dplyr::recode(mw_support_pre, "Strongly oppose raising the minimum wage"=4,"Somewhat oppose raising the minimum wage"=3,"Neither support nor oppose raising the minimum wage"=2,"Somewhat support raising the minimum wage"=1,"Strongly support raising the minimum wage"=0)/4,
    minwage_howhigh_pre = dplyr::recode(minwage_howhigh_pre, "Much lower than the current level"=4,"Somewhat lower than the current level"=3,"About the current level"=2,"Somewhat higher than the current level"=1,"Much higher than the current level"=0)/4,
    mw_help_pre_1 = dplyr::recode(mw_help_pre_1, "10"=9,"9"=8,"8"=7,"7"=6,"6"=5,"5"=4,"4"=3,"3"=2,"2"=1,"1"=0)/9,
    mw_restrict_pre_1 = dplyr::recode(mw_restrict_pre_1, "1"=9,"2"=8,"3"=7,"4"=6,"5"=5,"6"=4,"7"=3,"8"=2,"9"=1,"10"=0)/9,
    minwage_text_r_pre = (25-as.numeric(minwage_text_pre))/25,
  )
a$minwage_text_r_pre[as.numeric(a$minwage_text_pre)>25] <- NA


a <- a %>% 
  rowwise() %>%
  mutate(mw_index_pre = mean(c(minwage15_pre, rtwa_v1_pre,
                               rtwa_v2_pre, mw_support_pre, 
                               minwage_howhigh_pre, mw_help_pre_1,
                               mw_restrict_pre_1, minwage_text_r_pre), na.rm=T)) %>%
  ungroup()


# CRONBACH'S ALPHA
index_fa <- psych::alpha(select(a, minwage15_pre, rtwa_v1_pre, 
                                rtwa_v2_pre, mw_support_pre, minwage_howhigh_pre,
                                mw_help_pre_1, mw_restrict_pre_1, minwage_text_r_pre), check.keys = TRUE)

write.csv(data.frame(cor(select(a, minwage15_pre, rtwa_v1_pre, rtwa_v2_pre, 
                                mw_support_pre, minwage_howhigh_pre, mw_help_pre_1, 
                                mw_restrict_pre_1, minwage_text_r_pre), use = "complete.obs")),
row.names = T,file = "../results/cormat_mwindex_w1.csv")

# CORRELATION PLOT PRE-MINIMUM WAGE OPINION
pdf("corrplot_mwindex_w1.pdf")
w1_corrplot <- corrplot::corrplot(cor(select(a, minwage15_pre, rtwa_v1_pre, rtwa_v2_pre,
                                             mw_support_pre, minwage_howhigh_pre, mw_help_pre_1, 
                                             mw_restrict_pre_1, minwage_text_r_pre), 
                                      use = "complete.obs"),method = "shade")
dev.off()

(alpha <- index_fa$total["raw_alpha"]) # 0.9407615
writeLines(as.character(round(alpha,2)),con = "../results/outcomes_alpha_w1_mturk.tex",sep = "%")

tabyl(a,mw_index_pre)

##### MEDIA TRUST #####
a <- a %>%
  mutate( # higher = more trusting
    trust_majornews = dplyr::recode(info_trust_1,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    trust_localnews = dplyr::recode(info_trust_2,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    trust_social = dplyr::recode(info_trust_3,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    trust_youtube = dplyr::recode(info_trust_4,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    fabricate_majornews = dplyr::recode(mainstream_fakenews,"Never"=4,"Once in a while"=3,"About half the time"=2,"Most of the time"=1,"All the time"=0)/4,
    fabricate_youtube = dplyr::recode(youtube_fakenews,"Never"=4,"Once in a while"=3,"About half the time"=2,"Most of the time"=1,"All the time"=0)/4
  ) %>%
  rowwise() %>%
  mutate(media_trust = mean(trust_majornews,trust_localnews,fabricate_majornews,na.rm=T)) %>%
  ungroup()

media_trust_fa <- psych::alpha(select(a, trust_majornews,trust_localnews,fabricate_majornews), 
                               check.keys = TRUE)
(alpha <- media_trust_fa$total["raw_alpha"]) #. 0.7698292

##### AFFECTIVE POLARIZATION #####
a %>% 
  group_by(pid) %>% 
  summarize(mean_2=mean(as.numeric(political_lead_feels_2),na.rm=T), # Trump
            mean_5=mean(as.numeric(political_lead_feels_5),na.rm=T), # Biden
            mean_11=mean(as.numeric(political_lead_feels_11),na.rm=T), # dems
            mean_12=mean(as.numeric(political_lead_feels_12),na.rm=T)) # reps

a <- a %>%
  mutate( # higher = more trusting
    smart_dems = dplyr::recode(democrat_smart, "Extremely"=4,"Very"=3,"Somewhat"=2,"A little"=1,"Not at all"=0)/4,
    smart_reps = dplyr::recode(republican_smart, "Extremely"=4,"Very"=3,"Somewhat"=2,"A little"=1,"Not at all"=0)/4,
    comfort_dems = dplyr::recode(democrat_friends,"Extremely comfortable"=3,"Somewhat comfortable"=2,"Not too comfortable"=1,"Not at all comfortable"=0)/3,
    comfort_reps = dplyr::recode(republican_friends,"Extremely comfortable"=3,"Somewhat comfortable"=2,"Not too comfortable"=1,"Not at all comfortable"=0)/3,
    affpol_smart = case_when(
      pid==-1 ~ smart_dems-smart_reps,
      pid==1 ~ smart_reps-smart_dems
    ),
    affpol_comfort = case_when(
      pid==-1 ~ comfort_dems-comfort_reps,
      pid==1 ~ comfort_reps-comfort_dems
    )
  )

# Create a new variable 'thirds' based on attributes
a$thirds <- ifelse(!is.na(a$liberals_do) & is.na(a$moderates_do) & is.na(a$conservatives_do), 1, 
                   ifelse(is.na(a$liberals_do) & !is.na(a$moderates_do) & is.na(a$conservatives_do), 2, 
                          ifelse(is.na(a$liberals_do) & is.na(a$moderates_do) & !is.na(a$conservatives_do), 3, NA)))

tabyl(a$thirds)

#### OUTCOMES ####

##### POLICY OPINIONS ######
# convert to numeric unit scale:
a <- a %>%
  mutate( # higher = more pro-gun
    minwage15 = dplyr::recode(minwage15,"Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    rtwa_v1 = dplyr::recode(rtwa_v1_updated, "Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    rtwa_v2 = dplyr::recode(rtwa_v2_updated, "Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    mw_support = dplyr::recode(mw_support, "Strongly oppose raising the minimum wage"=4,"Somewhat oppose raising the minimum wage"=3,"Neither support nor oppose raising the minimum wage"=2,"Somewhat support raising the minimum wage"=1,"Strongly support raising the minimum wage"=0)/4,
    minwage_howhigh = dplyr::recode(minwage_howhigh, "Much lower than the current level"=4,"Somewhat lower than the current level"=3,"About the current level"=2,"Somewhat higher than the current level"=1,"Much higher than the current level"=0)/4,
    mw_help_1 = dplyr::recode(mw_help_1, "10"=9,"9"=8,"8"=7,"7"=6,"6"=5,"5"=4,"4"=3,"3"=2,"2"=1,"1"=0)/9,
    mw_restrict_1 = dplyr::recode(mw_restrict_1, "1"=9,"2"=8,"3"=7,"4"=6,"5"=5,"6"=4,"7"=3,"8"=2,"9"=1,"10"=0)/9,
    minwage_text_r = (25-as.numeric(minwage_text))/25,
  )
a$minwage_text_r[as.numeric(a$minwage_text)>25] <- NA

a <- a %>% 
  rowwise() %>%
  mutate(mw_index = mean(c(minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, 
                           mw_help_1, mw_restrict_1, minwage_text_r), na.rm=T)) %>%
  ungroup()

# CRONBACH-S ALPHA
index_fa <- psych::alpha(select(a, minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, 
                                mw_help_1, mw_restrict_1, minwage_text_r), check.keys = T)

write.csv(data.frame(cor(select(a, minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, 
                                mw_help_1, mw_restrict_1, minwage_text_r), use = "complete.obs")),
          row.names = T,file = "../results/cormat_mw_index_w2.csv")

pdf("corrplot_mwindex_w2.pdf")
a_corrplot <- corrplot::corrplot(cor(select(a, minwage15, rtwa_v1, rtwa_v2, mw_support, 
                                            minwage_howhigh, mw_help_1, mw_restrict_1, minwage_text_r), 
                                     use = "complete.obs"),method = "shade")
dev.off()

(alpha <- index_fa$total["raw_alpha"]) # 0.9582061

### SURVEY PREPROCESSING DONE ###

## YTRECS SESSION DATA -------------------------------------------------------
ytrecs <- read_rds("../data/shorts/ytrecs_sessions_may2024.rds") %>%
  clean_names() %>%
  as_tibble()

## EXTRACTING TOPICID AND URLID
a <- a %>%
  ungroup() %>%
  mutate(
    topic_id = str_extract(video_link, "topicid=([a-z]{2}[1-6])") %>% str_replace("topicid=", ""),
    urlid = str_extract(video_link, "id=(mt_\\d+)") %>% str_replace("id=", "")
  )

## USING THE FIRST SESSION AS THE VALID ONE IF A PERSON HAS MULTIPLE ATTEMPTS
ytrecs <- ytrecs %>%
  group_by(topic_id, urlid) %>% 
  mutate(dupes = n(),
         first_session = ifelse(row_number() == 1, 1, 0)
  ) %>%
  filter(first_session == 1) # using the first session as valid one

a <- left_join(a, ytrecs,by=c("topic_id","urlid"))

## EXTRACTING TREATMENT ARM
extract_treatmentarm <- function(url) {
  pattern <- "topicid=([a-z]{2})" #[a-z]{2}[1-6]
  match <- str_match(url, pattern)
  if (!is.na(match[2])) {
    return(match[2])
  } else {
    return(NA)
  }
}

# APPLY THE FUNCTION TO THE VIDEO_LINK COLUMN
a <- a %>%
  rowwise() %>%
  mutate(treatment_arm = extract_treatmentarm(video_link)) %>%
  ungroup()

write_csv(a, "../results/intermediate data/shorts/qualtrics_w12_clean_ytrecs_may2024.csv")
rm(list = ls())

### PREPROCESSING DONE ----------------------
