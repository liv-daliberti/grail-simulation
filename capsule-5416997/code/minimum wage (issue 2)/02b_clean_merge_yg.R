cat(rep('=', 80),
    '\n\n',
    'OUTPUT FROM: minimum wage (issue 2)/02b_clean_merge_yg.R',
    '\n\n',
    sep = ''
    )

library(tidyverse)
library(lubridate)
library(stargazer)
library(haven)
library(janitor)

yg <- read_sav("../data/minimum wage (issue 2)/PRIN0016_MERGED_OUTPUT.sav")

## Recodes:
yg <- yg %>% mutate(start_date = as_datetime(starttime),
                    end_date = as_datetime(endtime),
                    start_date_w2 = as_datetime(starttime_W2),
                    end_date_w2 = as_datetime(endtime_W2),
                    survey_time = as.numeric(end_date-start_date),
                    survey_time_w2 = as.numeric(end_date_w2-start_date_w2),
)

print('wave 1 survey time')
summary(yg$survey_time)

print('wave 2 survey time')
summary(yg$survey_time_w2)

#### Demographics ####
yg <- yg %>%
  mutate(female = ifelse(gender4 == 2, 1, 0),
         male = ifelse(gender4 == 1, 1, 0),
         black = ifelse(race == 2, 1, 0),
         white = ifelse(race == 1, 1, 0),
         college = ifelse(educ == 5 | educ == 6, 1, 0),
         income_gt50k = ifelse(faminc_new >= 6 & faminc_new <= 16, 1, 0)
  )

# PID:
yg <- yg %>%
  mutate(pid = case_when(pid3==1 ~ -1,
                         pid3==2 ~ 1,
                         pid7>4 & pid7<8 ~ 1,
                         pid7<4 ~ -1,
                         pid7==4 ~ 0))

yg <- yg %>%
  mutate(ideo = case_when(ideo5<3 ~ -1,
                          ideo5>3 & ideo5<6 ~ 1,
                          ideo5==3 ~ 0))

yg$age <- 2022 - yg$birthyr

# age categories: 18-29;  30-44;  45-64;  65+
yg <- yg %>%
  mutate(age_cat = case_when(age>=18 & age<=29 ~ "18-29",
                             age>=30 & age<=44 ~ "30-44",
                             age>=45 & age<=64 ~ "45-64",
                             age>=65 ~ "65+"
  ))

yg <- yg %>%
  fastDummies::dummy_cols(select_columns = "age_cat",remove_selected_columns = F)

yg <- yg %>%
  mutate(pol_interest = ifelse(newsint>4,NA_real_,newsint),
         pol_interest = (4-pol_interest)/3,
         youtube_freq_v2 = ifelse(youtube_freq>10,NA_real_,youtube_freq),
         freq_youtube_v2 = 10-youtube_freq_v2,
         freq_youtube = (Q77-1)
  )


# Descriptives ------------------------------------------------------------

descr_data <- as.data.frame(select(yg,
                     female,
                     white,
                     black,
                     age,
                     college,
                     income_gt50k))
descr_data <- descr_data %>% filter(rowSums(is.na(.)) != ncol(.))
descriptive_tab <- stargazer(descr_data,
                             summary = T, digits=2,
                             summary.stat=c("mean","sd","median","min","max","n"),
                             covariate.labels = c("Female",
                                                  "White",
                                                  "Black",
                                                  "Age",
                                                  "College educated",
                                                  "Income \\textgreater 50k"),
                             float = F,
                             out = "../results/minwage_descriptive_tab_yg.tex")

summary_tab <- yg %>%
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
                                                           "White"
  ),ordered=T))

(descrip_fig <- ggplot(summary_tab) +
    geom_point(aes(y=outcome_pretty,x=survey_avg)) +
    geom_text(aes(y=outcome_pretty,x=survey_avg,label=paste0(round(100*survey_avg,0),"%")),nudge_x = 0.1) +
    scale_y_discrete("") +
    scale_x_continuous("",labels=scales::percent_format(),limits=c(0,1)) +
    theme_bw()
)

ggsave(descrip_fig,filename = "../results/minwage_demographics_yg.pdf",height=5,width=4)



#### A/V check
print('audio ok:')
length(which(yg$Q81_W2 == 1))/length(which(!is.na(yg$Q81_W2)))
print('video ok:')
length(which(yg$Q82_W2 == 1))/length(which(!is.na(yg$Q82_W2)))



#### Outcomes ####

##### policy opinions #####
# convert to numeric unit scale:
yg <- yg %>%
  mutate( # higher = more conservative or anti-min wage
    minwage15_w1 = (minwage15-1)/4,
    rtwa_v1_w1 = (RTWA_v1-1)/4,
    rtwa_v2_w1 = (RTWA_v2-1)/4,
    mw_support_w1 = (mw_support-1)/4,
    minwage_howhigh_w1 = (minwage_howhigh-1)/4,
    mw_help_w1 = (mw_help_a-1)/9,
    mw_restrict_w1 = (10-mw_restrict_1)/9,
    minwage_text_r_w1 = (25-as.numeric(minwage_text))/25,
  )

yg <- yg %>%
  rowwise() %>%
  mutate(mw_index_w1 = mean(c(minwage15_w1, rtwa_v1_w1, rtwa_v2_w1, mw_support_w1, minwage_howhigh_w1, mw_help_w1, mw_restrict_w1, minwage_text_r_w1), na.rm=T)) %>%
  ungroup()

# Cronbach's alpha
index_fa <- psych::alpha(select(yg, minwage15_w1, rtwa_v1_w1, rtwa_v2_w1, mw_support_w1, minwage_howhigh_w1, mw_help_w1, mw_restrict_w1, minwage_text_r_w1), check.keys = TRUE)
write.csv(data.frame(cor(select(yg, minwage15_w1, rtwa_v1_w1, rtwa_v2_w1, mw_support_w1, minwage_howhigh_w1, mw_help_w1, mw_restrict_w1, minwage_text_r_w1), use = "complete.obs")),row.names = T,
          file = "../results/cormat_mwindex_w1_yg.csv")

pdf("../results/corrplot_mwindex_w1_yg.pdf")
w1_corrplot <- corrplot::corrplot(cor(select(yg, minwage15_w1, rtwa_v1_w1, rtwa_v2_w1, mw_support_w1, minwage_howhigh_w1, mw_help_w1, mw_restrict_w1, minwage_text_r_w1), use = "complete.obs"),method = "shade")
dev.off()

alpha <- index_fa$total["raw_alpha"]
writeLines(as.character(round(alpha,2)),con = "../results/minwage_outcomes_alpha_w1_yg.tex",sep = "%")

# FACTOR ANALYSIS WITH VARIMAX ROTATION (PRE)
pca2 <- psych::principal(select(yg, minwage15_w1, rtwa_v1_w1, rtwa_v2_w1, mw_support_w1, minwage_howhigh_w1, mw_help_w1, mw_restrict_w1, minwage_text_r_w1),
                         rotate="varimax",
                         nfactors=1
)
pc2 <- pca2$Vaccounted[2]
writeLines(as.character(round(pc2, 2)),con = "../results/outcomes_pc2_study3_pre.tex",sep = "%")

##### media trust #####
yg <- yg %>%
  mutate( # higher = more trusting
    trust_majornews_w1 = (4-Q58_a)/3,
    trust_localnews_w1 = (4-Q58_b)/3,
    trust_social_w1 = (4-Q58_c)/3,
    trust_youtube_w1 = (4-Q58_d)/3,
    fabricate_majornews_w1 = (5-Q89b)/4,
    fabricate_youtube_w1 = (5-Q90)/4
  ) %>%
  rowwise() %>%
  mutate(media_trust_w1 = mean(trust_majornews_w1,trust_localnews_w1,fabricate_majornews_w1,na.rm=T)) %>%
  ungroup()

media_trust_fa <- psych::alpha(select(yg, trust_majornews_w1,trust_localnews_w1,fabricate_majornews_w1), check.keys = TRUE)
print('media trust alpha:')
media_trust_fa$total["raw_alpha"]


##### affective polarization #####
# check FTs:
yg %>%
  group_by(pid) %>%
  summarize(mean_2=mean(as.numeric(Q5_a),na.rm=T), # Trump
            mean_5=mean(as.numeric(Q5_b),na.rm=T), # Biden
            mean_11=mean(as.numeric(Q5_c),na.rm=T), # dems
            mean_12=mean(as.numeric(Q5_d),na.rm=T)) # reps

yg <- yg %>%
  mutate(
    smart_dems = (5-Q61)/4,
    smart_reps = (5-Q62)/4,
    comfort_dems = (Q87b-1)/3,
    comfort_reps = (Q88-1)/3,
    ft_dems = as.numeric(Q5_c),
    ft_reps = as.numeric(Q5_d),
    affpol_smart = case_when(
      pid==-1 ~ smart_dems-smart_reps,
      pid==1 ~ smart_reps-smart_dems
    ),
    affpol_comfort = case_when(
      pid==-1 ~ comfort_dems-comfort_reps,
      pid==1 ~ comfort_reps-comfort_dems
    ),
    affpol_ft = case_when(
      pid==-1 ~ ft_dems-ft_reps,
      pid==1 ~ ft_reps-ft_dems
    )
  )



# W2 ----------------------------------------------------------------------

##### policy opinions #####
# convert to numeric unit scale:
yg <- yg %>%
  mutate( # higher = more conservative or anti-min wage
    minwage15_w2 = (minwage15_W2-1)/4,
    rtwa_v1_w2 = (RTWA_v1_W2-1)/4,
    rtwa_v2_w2 = (RTWA_v2_W2-1)/4,
    mw_support_w2 = (mw_support_W2-1)/4,
    minwage_howhigh_w2 = (minwage_howhigh_W2-1)/4,
    mw_help_w2 = (mw_help_a_W2-1)/9,
    mw_restrict_w2 = (10-mw_restrict_1_W2)/9,
    minwage_text_r_w2 = (25-as.numeric(minwage_text_W2))/25,
  )

yg <- yg %>%
  rowwise() %>%
  mutate(mw_index_w2 = mean(c(minwage15_w2, rtwa_v1_w2, rtwa_v2_w2, mw_support_w2, minwage_howhigh_w2, mw_help_w2, mw_restrict_w2, minwage_text_r_w2), na.rm=T)) %>%
  ungroup()

# Cronbach's alpha
index_fa <- psych::alpha(select(yg, minwage15_w2, rtwa_v1_w2, rtwa_v2_w2, mw_support_w2, minwage_howhigh_w2, mw_help_w2, mw_restrict_w2, minwage_text_r_w2), check.keys = TRUE)
write.csv(data.frame(cor(select(yg, minwage15_w2, rtwa_v1_w2, rtwa_v2_w2, mw_support_w2, minwage_howhigh_w2, mw_help_w2, mw_restrict_w2, minwage_text_r_w2), use = "complete.obs")),row.names = T,
          file = "../results/cormat_mwindex_w2_yg.csv")

pdf("../results/corrplot_mwindex_w2_yg.pdf")
w2_corrplot <- corrplot::corrplot(cor(select(yg, minwage15_w2, rtwa_v1_w2, rtwa_v2_w2, mw_support_w2, minwage_howhigh_w2, mw_help_w2, mw_restrict_w2, minwage_text_r_w2), use = "complete.obs"),method = "shade")
dev.off()

print('wave 2 policy opinion alpha:')
(alpha <- index_fa$total["raw_alpha"])
writeLines(as.character(round(alpha,2)),con = "../results/minwage_outcomes_alpha_w2_mturk.tex",sep = "%")

# FACTOR ANALYSIS WITH VARIMAX ROTATION (POST)
pca2 <- psych::principal(select(yg, minwage15_w2, rtwa_v1_w2, rtwa_v2_w2, mw_support_w2, minwage_howhigh_w2, mw_help_w2, mw_restrict_w2, minwage_text_r_w2),
                         rotate="varimax",
                         nfactors=1
)
pc2 <- pca2$Vaccounted[2]
writeLines(as.character(round(pc2, 2)),con = "../results/outcomes_pc2_study3_post.tex",sep = "%")

##### media trust #####
yg <- yg %>%
  mutate( # higher = more trusting
    trust_majornews_w2 = (4-Q58_a_W2)/3,
    trust_localnews_w2 = (4-Q58_b_W2)/3,
    trust_social_w2 = (4-Q58_c_W2)/3,
    trust_youtube_w2 = (4-Q58_d_W2)/3,
    fabricate_majornews_w2 = (5-Q89b_W2)/4,
    fabricate_youtube_w2 = (5-Q90_W2)/4
  ) %>%
  rowwise() %>%
  mutate(media_trust_w2 = mean(c(trust_majornews_w2,trust_localnews_w2,fabricate_majornews_w2),na.rm=T)) %>%
  ungroup()

##### affective polarization #####
print('check affpol feeling thermometers:')
yg %>%
  group_by(pid) %>%
  summarize(mean_2=mean(as.numeric(Q5_a_W2),na.rm=T), # Trump
            mean_5=mean(as.numeric(Q5_b_W2),na.rm=T), # Biden
            mean_11=mean(as.numeric(Q5_c_W2),na.rm=T), # dems
            mean_12=mean(as.numeric(Q5_d_W2),na.rm=T)) # reps

yg <- yg %>%
  mutate( # higher = more trusting
    smart_dems_w2 = (5-Q61_W2)/4,
    smart_reps_w2 = (5-Q62_W2)/4,
    comfort_dems_w2 = (Q92_W2-1)/3,
    comfort_reps_w2 = (Q94_W2-1)/3,
    ft_dems_w2 = as.numeric(Q5_c_W2),
    ft_reps_w2 = as.numeric(Q5_d_W2),
    affpol_smart_w2 = case_when(
      pid==-1 ~ smart_dems_w2-smart_reps_w2,
      pid==1 ~ smart_reps_w2-smart_dems_w2
    ),
    affpol_comfort_w2 = case_when(
      pid==-1 ~ comfort_dems_w2-comfort_reps_w2,
      pid==1 ~ comfort_reps_w2-comfort_dems_w2
    ),
    affpol_ft_w2 = case_when(
      pid==-1 ~ ft_dems_w2-ft_reps_w2,
      pid==1 ~ ft_reps_w2-ft_dems_w2
    )
  )


## YTRecs session data: -------------------------------------------------------

ytrecs <- read_rds("../data/minimum wage (issue 2)/min_wage_data.rds") %>%
  clean_names() %>%
  as_tibble()

ytrecs <- ytrecs %>%
  mutate(duration = end_time2 - start_time2) %>%
  select(topic_id,urlid,pro,anti,duration,pro_up,pro_down,anti_up,anti_down,pro_save,anti_save,start_time2, end_time2) %>%
  filter(str_detect(urlid,"mt_",negate = T) & !is.na(pro))

ytrecs <- ytrecs %>%
  group_by(topic_id,urlid) %>%
  mutate(dupes = n(),
         max_duration = ifelse(duration==max(duration),1,0)
  ) %>%
  filter(max_duration==1) # using longest session as valid one

ytrecs <- ytrecs %>%
  mutate(
    pro_up = replace_na(pro_up,0),
    pro_down = replace_na(pro_down,0),
    anti_up = replace_na(anti_up,0),
    anti_down = replace_na(anti_down,0),
    pro_save = replace_na(pro_save,0),
    anti_save = replace_na(anti_save,0)) %>%
  rowwise() %>%
  mutate(total_likes = sum(pro_up,anti_up,na.rm=T),
         total_dislikes = sum(pro_down,anti_down,na.rm=T),
         total_thumbs = sum(pro_up,pro_down,anti_up,anti_down,na.rm=T),
         total_saved = sum(pro_save,anti_save,na.rm=T),
         total_interactions = sum(pro_up,pro_down,anti_up,anti_down,pro_save,anti_save,na.rm=T),
         positive_interactions = total_likes + total_saved - total_dislikes
  )

ytrecs <- ytrecs %>%
  mutate(seed = str_replace(topic_id,".*_(\\w+)$","\\1")) %>%
  mutate(pro = as.numeric(pro),
         anti = as.numeric(anti)) %>%
  mutate(pro_fraction_chosen = case_when(
    seed=="anti" ~ pro/(pro+anti-1),
    seed=="pro" ~ (pro-1)/(pro+anti-1)
  ))
# adjust for zeros:
ytrecs$pro_fraction_chosen[ytrecs$pro==0 & ytrecs$anti==0] <- NA


yg <- yg %>%
  ungroup() %>%
  mutate(
    urlid = session_visa_W2
  )

yg <- left_join(yg,ytrecs,by=c("urlid"))

print("ISSUE 2 NUMBERS:")
print(paste('count w/ valid ytrecs data:', sum(!is.na(yg$pro))))
print(paste('count w/ valid ytrecs interactions:', sum(!is.na(yg$total_thumbs))))
print('interactions:')
summary(yg$total_interactions)

# create numeric dosage version of treatment:
yg <- yg %>%
  mutate(treatment_arm = haven::as_factor(treatment_arm_W2),
         treatment_dose = dplyr::recode(treatment_arm,
                                 "anti_31"= 1, "anti_22" = 0,
                                 "pro_31"= 1, "pro_22" = 0,
                                 "control"=NA_real_),
         treatment_seed = str_replace(treatment_arm,"(.*)\\_\\d{2}","\\1")
  )

terciles <- read_csv("../results/intermediate data/minimum wage (issue 2)/yougov_terciles.csv")
yg <- left_join(yg,select(terciles,caseid,thirds=tercile),by="caseid")

write_csv(yg, "../results/intermediate data/minimum wage (issue 2)/yg_w12_clean.csv")
