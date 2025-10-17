## YouTube Algorithms and Minimum Wage Opinions
## Data collected April 2022 via MTurk/CloudResearch

cat(rep('=', 80),
    '\n\n',
    'OUTPUT FROM: minimum wage (issue 2)/02_clean_merge.R',
    '\n\n',
    sep = ''
    )

## Preamble ----------------------------
library(tidyverse)
library(janitor)
library(lubridate)
library(stargazer)
library(broom)

a <- read_csv("../data/minimum wage (issue 2)/YouTube+Min+Wage+-+Apr+2022+presurvey_July+5,+2022_12.50.csv")[-c(1,2),] %>%
  clean_names()


# Wave 1 =======================================================================

## Recodes:
a <- a %>% mutate(start_date = as_datetime(start_date),
                  end_date = as_datetime(end_date),
                  survey_time = as.numeric(end_date-start_date))

print('wave 1 survey time')
summary(a$survey_time)

# Demographics -----------------------------------------------------------------

a <- a %>%
  mutate(female = ifelse(q26 == "Woman", 1, 0),
         male = ifelse(q26 == "Man", 1, 0),
         black = ifelse(str_detect(q29, "Black"), 1, 0),
         white = ifelse(str_detect(q29, "White"), 1, 0),
         college = ifelse(str_detect(q30, "college ") | str_detect(q30, "Post"), 1, 0),
         income_gt50k = ifelse(q31 %in% names(table(a$q31))[c(2,3,5,10:13)], 1, 0)
  )
a$income_gt50k[is.na(a$q31)] <- NA

# PID:

a <- a %>%
  mutate(pid = case_when(pid1=="Democrat" ~ -1,
                         pid1=="Republican" ~ 1,
                         pid4=="Closer to the Republican Party" ~ 1,
                         pid4=="Closer to the Democratic Party" ~ -1,
                         pid4=="Neither" ~ 0))

tabyl(a,pid)


a <- a %>%
  mutate(ideo = case_when(ideo1=="Liberal" ~ -1,
                          ideo1=="Conservative" ~ 1,
                          ideo4=="Closer to conservatives" ~ 1,
                          ideo4=="Closer to liberals" ~ -1,
                          ideo4=="Neither" ~ 0))

tabyl(a,ideo)

a$age <- 2022-as.numeric(a$q27)

# age categories: 18-29;  30-44;  45-64;  65+
a <- a %>%
  mutate(age_cat = case_when(age>=18 & age<=29 ~ "18-29",
                             age>=30 & age<=44 ~ "30-44",
                             age>=45 & age<=64 ~ "45-64",
                             age>=65 ~ "65+"
  ))
a <- a %>%
  fastDummies::dummy_cols(select_columns = "age_cat",remove_selected_columns = F)

## Need:
# political  interest  (5-point  scale:   1=Not  atall interested, 5=Extremely interested),
# self-reported YouTube usage frequency (7-pointscale: 0=None, 6=More than 3 hours per day),
# number of self-reported favorite YouTubechannels (count coded from open-ended question: “Who/what are your favorite YouTubebroadcasters or channels?”; 0 if blank),
# indicator for having watched videos from popularchannels (1 if any selected:  “In the past week, have you watched videos from any of thefollowing  YouTube  broadcasters  or  channels?”),
# video  vs.   text  preference  (1=Alwaysprefer  videos,  10=Always  prefer  text),
# gun  enthusiasm  (additive  index  of  “Do  you  ordoes anyone in your household own a gun?”  with yes=1 and “How often, if ever, do youvisit websites about guns, hunting or other shooting sports?”  from 0=Never or Hardlyever to 1=Sometimes or Often),
# gun policy issue importance (4-point scale:  1=Not atall important, 4=Very important)

a <- a %>%
  mutate(pol_interest = dplyr::recode(q91,"Extremely interested"=5,"Very interested"=4,"Somewhat interested"=3,"Not very interested"=2,"Not at all interested"=1),
         freq_youtube = dplyr::recode(q77,"More than 3 hours per day"=6,"2–3 hours per day"=5,"1–2 hours per day"=4,"31–59 minutes per day"=3,"10–30 minutes per day"=2,"Less than 10 minutes per day"=1,"None"=0),
  )

descr_data <- as.data.frame(select(a,
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
                             out = "../results/minwage_descriptive_tab.tex")

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
                                                           "White"
  ),ordered=T))

(descrip_fig <- ggplot(summary_tab) +
    geom_point(aes(y=outcome_pretty,x=survey_avg)) +
    geom_text(aes(y=outcome_pretty,x=survey_avg,label=paste0(round(100*survey_avg,0),"%")),nudge_x = 0.1) +
    scale_y_discrete("") +
    scale_x_continuous("",labels=scales::percent_format(),limits=c(0,1)) +
    theme_bw()
)
ggsave(descrip_fig,filename = "../results/minwage_demographics.pdf",height=5,width=4)



# A/V check
print('audio ok:')
length(which(a$q87 == "Quick and easy"))/length(a$q87)
print('video ok:')
length(which(a$q89 == "wikiHow"))/length(a$q89)#dk

#### Outcomes ####

##### policy opinions #####
# convert to numeric unit scale:
a <- a %>%
  mutate( # higher = more conservative or anti-min wage
    minwage15 = dplyr::recode(minwage15,"Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    rtwa_v1 = dplyr::recode(rtwa_v1, "Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    # minwage_inflation = recode(minwage_inflation,"Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    rtwa_v2 = dplyr::recode(rtwa_v2, "Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    mw_support = dplyr::recode(mw_support, "Strongly oppose raising the minimum wage"=4,"Somewhat oppose raising the minimum wage"=3,"Neither support nor oppose raising the minimum wage"=2,"Somewhat support raising the minimum wage"=1,"Strongly support raising the minimum wage"=0)/4,
    minwage_howhigh = dplyr::recode(minwage_howhigh, "Much lower than the current level"=4,"Somewhat lower than the current level"=3,"About the current level"=2,"Somewhat higher than the current level"=1,"Much higher than the current level"=0)/4,
    mw_help = dplyr::recode(mw_help, "Would hurt low-income workers\n10\n"=9,"9"=8,"8"=7,"7"=6,"6"=5,"5"=4,"4"=3,"3"=2,"2"=1,"Would help low-income workers\n1"=0)/9,
    mw_restrict = dplyr::recode(mw_restrict, "Would restrict businesses' freedom\n1\n"=9,"2"=8,"3"=7,"4"=6,"5"=5,"6"=4,"7"=3,"8"=2,"9"=1,"Would protect workers from exploitation\n10\n"=0)/9,
    minwage_text_r = (25-as.numeric(minwage_text))/25,
  )
a$minwage_text_r[as.numeric(a$minwage_text)>25] <- NA

a <- a %>%
  rowwise() %>%
  mutate(mw_index = mean(c(minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text_r), na.rm=T)) %>%
  ungroup()

# Cronbach's alpha
index_fa <- psych::alpha(select(a, minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text_r), check.keys = TRUE)
write.csv(data.frame(cor(select(a, minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text_r), use = "complete.obs")),row.names = T,
          file = "../results/minwage_cormat_mwindex_w1.csv")

pdf("../results/corrplot_mwindex_w1.pdf")
w1_corrplot <- corrplot::corrplot(cor(select(a, minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text_r), use = "complete.obs"),method = "shade")
dev.off()

alpha <- index_fa$total["raw_alpha"]
writeLines(as.character(round(alpha,2)),con = "../results/minwage_outcomes_alpha_w1_mturk.tex",sep = "%")

# FACTOR ANALYSIS WITH VARIMAX ROTATION (PRE)
pca2 <- psych::principal(select(a, minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text_r),
                         rotate="varimax",
                         nfactors=1
)
pc2 <- pca2$Vaccounted[2]
writeLines(as.character(round(pc2, 2)),con = "../results/outcomes_pc2_study2_pre.tex",sep = "%")

##### media trust #####
a <- a %>%
  mutate( # higher = more trusting
    trust_majornews = dplyr::recode(q58_1,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    trust_localnews = dplyr::recode(q58_2,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    trust_social = dplyr::recode(q58_3,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    trust_youtube = dplyr::recode(q58_4,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    fabricate_majornews = dplyr::recode(q89_1,"Never"=4,"Once in a while"=3,"About half the time"=2,"Most of the time"=1,"All the time"=0)/4,
    fabricate_youtube = dplyr::recode(q90,"Never"=4,"Once in a while"=3,"About half the time"=2,"Most of the time"=1,"All the time"=0)/4
  ) %>%
  rowwise() %>%
  mutate(media_trust = mean(trust_majornews,trust_localnews,fabricate_majornews,na.rm=T)) %>%
  ungroup()

media_trust_fa <- psych::alpha(select(a, trust_majornews,trust_localnews,fabricate_majornews), check.keys = TRUE)
print('media trust alpha:')
media_trust_fa$total["raw_alpha"]


##### affective polarization #####
# check FTs:
a %>%
  group_by(pid) %>%
  summarize(mean_2=mean(as.numeric(q5_2),na.rm=T), # Trump
            mean_5=mean(as.numeric(q5_5),na.rm=T), # Biden
            mean_11=mean(as.numeric(q5_11),na.rm=T), # dems
            mean_12=mean(as.numeric(q5_12),na.rm=T)) # reps

a <- a %>%
  mutate( # higher = more trusting
    smart_dems = dplyr::recode(q61, "Extremely"=4,"Very"=3,"Somewhat"=2,"A little"=1,"Not at all"=0)/4,
    smart_reps = dplyr::recode(q62_1, "Extremely"=4,"Very"=3,"Somewhat"=2,"A little"=1,"Not at all"=0)/4,
    comfort_dems = dplyr::recode(q87_1,"Extremely comfortable"=3,"Somewhat comfortable"=2,"Not too comfortable"=1,"Not at all comfortable"=0)/3,
    comfort_reps = dplyr::recode(q88,"Extremely comfortable"=3,"Somewhat comfortable"=2,"Not too comfortable"=1,"Not at all comfortable"=0)/3,
    ft_dems = as.numeric(q5_11),
    ft_reps = as.numeric(q5_12),
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



## for reinvitations:
w1_reinvited <- a %>% filter(q87 == "Quick and easy", q89 == "wikiHow") # AV checks
w1_reinvited <- w1_reinvited %>% filter(mw_index >= 0.025, mw_index <= 0.975)


w1_reinvited$thirds <- cut(w1_reinvited$mw_index, breaks = quantile(w1_reinvited$mw_index, c(0, 1/3, 2/3, 1)), include.lowest = TRUE, labels = 1:3)
a$thirds <- w1_reinvited$thirds[match(a$worker_id,w1_reinvited$worker_id)]

write_csv(a, "../results/intermediate data/minimum wage (issue 2)/qualtrics_w1_clean.csv")


# Wave 2 (main survey) =========================================================


w2 <- read_csv("../data/minimum wage (issue 2)/YouTube+Min+Wage+-+Apr+2022+main+survey_July+5,+2022_12.47.csv")[-c(1,2),] %>%
  clean_names() %>%
  select(-thirds) # remove all-NA column

w2 <- w2 %>% mutate(start_date_w2 = as_datetime(start_date),
                    end_date_w2 = as_datetime(end_date),
                    survey_time_w2 = as.numeric(end_date_w2-start_date_w2))

print('wave 2 survey time:')
summary(w2$survey_time_w2)

print('audio ok:')
length(which(w2$q81 == "Quick and easy"))/length(w2$q81)
print('video ok:')
length(which(w2$q82 == "wikiHow"))/length(w2$q82)


#### Outcomes ####

##### policy opinions ######
# convert to numeric unit scale:
w2 <- w2 %>%
  mutate( # higher = more pro-gun
    minwage15 = dplyr::recode(minwage15,"Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    rtwa_v1 = dplyr::recode(rtwa_v1, "Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    rtwa_v2 = dplyr::recode(rtwa_v2, "Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    mw_support = dplyr::recode(mw_support, "Strongly oppose raising the minimum wage"=4,"Somewhat oppose raising the minimum wage"=3,"Neither support nor oppose raising the minimum wage"=2,"Somewhat support raising the minimum wage"=1,"Strongly support raising the minimum wage"=0)/4,
    minwage_howhigh = dplyr::recode(minwage_howhigh, "Much lower than the current level"=4,"Somewhat lower than the current level"=3,"About the current level"=2,"Somewhat higher than the current level"=1,"Much higher than the current level"=0)/4,
    mw_help = dplyr::recode(mw_help, "Would hurt low-income workers\n10\n"=9,"9"=8,"8"=7,"7"=6,"6"=5,"5"=4,"4"=3,"3"=2,"2"=1,"Would help low-income workers\n1"=0)/9,
    mw_restrict = dplyr::recode(mw_restrict, "Would restrict businesses' freedom\n1\n"=9,"2"=8,"3"=7,"4"=6,"5"=5,"6"=4,"7"=3,"8"=2,"9"=1,"Would protect workers from exploitation\n10\n"=0)/9,
    minwage_text_r = (25-as.numeric(minwage_text))/25,
  )
w2$minwage_text_r[as.numeric(w2$minwage_text)>25] <- NA

w2 <- w2 %>%
  rowwise() %>%
  mutate(mw_index = mean(c(minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text_r), na.rm=T)) %>%
  ungroup()


# Cronbach's alpha
index_fa <- psych::alpha(select(w2, minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text_r), check.keys = T)
write.csv(data.frame(cor(select(w2, minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text_r), use = "complete.obs")),row.names = T,
          file = "../results/minwage_cormat_mw_index_w2.csv")

pdf("../results/minwage_corrplot_mwindex_w2.pdf")
w2_corrplot <- corrplot::corrplot(cor(select(w2, minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text_r), use = "complete.obs"),method = "shade")
dev.off()

print('wave 2 policy opinion alpha:')
(alpha <- index_fa$total["raw_alpha"])
writeLines(as.character(round(alpha,2)),con = "../results/minwage_outcomes_alpha_w2_mturk.tex",sep = "%")

# FACTOR ANALYSIS WITH VARIMAX ROTATION (POST)
pca2 <- psych::principal(select(w2, minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text_r),
                         rotate="varimax",
                         nfactors=1
)
pc2 <- pca2$Vaccounted[2]
writeLines(as.character(round(pc2, 2)),con = "../results/outcomes_pc2_study2_post.tex",sep = "%")

##### media trust #####
w2 <- w2 %>%
  mutate( # higher = more trusting
    trust_majornews = dplyr::recode(q96_1,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    trust_localnews = dplyr::recode(q96_2,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    trust_social = dplyr::recode(q96_3,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    trust_youtube = dplyr::recode(q96_4,"A lot"=3,"Some"=2,"Not too much"=1,"Not at all"=0)/3,
    fabricate_majornews = dplyr::recode(q98,"Never"=4,"Once in a while"=3,"About half the time"=2,"Most of the time"=1,"All the time"=0)/4,
    fabricate_youtube = dplyr::recode(q100_1,"Never"=4,"Once in a while"=3,"About half the time"=2,"Most of the time"=1,"All the time"=0)/4
  ) %>%
  rowwise() %>%
  mutate(media_trust = mean(trust_majornews,trust_localnews,fabricate_majornews,na.rm=T)) %>%
  ungroup()

##### affective polarization #####
print('check affpol feeling thermometers:')
w2 <- w2 %>%
  mutate(
    smart_dems = dplyr::recode(q61, "Extremely"=4,"Very"=3,"Somewhat"=2,"A little"=1,"Not at all"=0)/4,
    smart_reps = dplyr::recode(q62_1, "Extremely"=4,"Very"=3,"Somewhat"=2,"A little"=1,"Not at all"=0)/4,
    comfort_dems = dplyr::recode(q92,"Extremely comfortable"=3,"Somewhat comfortable"=2,"Not too comfortable"=1,"Not at all comfortable"=0)/3,
    comfort_reps = dplyr::recode(q94,"Extremely comfortable"=3,"Somewhat comfortable"=2,"Not too comfortable"=1,"Not at all comfortable"=0)/3,
    ft_dems = as.numeric(q90_11),
    ft_reps = as.numeric(q90_12)
  )

write_csv(w2, "../results/intermediate data/minimum wage (issue 2)/qualtrics_w2_clean.csv")


# join to W1 by MT worker ID:
w12 <- left_join(a, filter(w2,!is.na(worker_id)), by = "worker_id",suffix=c("_w1","_w2"))
names(w12)

w12 <- w12 %>%
  mutate(
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
         ))

write_csv(w12, "../results/intermediate data/minimum wage (issue 2)/qualtrics_w12_clean.csv")


## YTRecs session data: -------------------------------------------------------

ytrecs <- read_rds("../data/minimum wage (issue 2)/min_wage_data.rds") %>%
  clean_names() %>%
  as_tibble()

ytrecs <- ytrecs %>%
  mutate(duration = end_time2 - start_time2) %>%
  select(topic_id,urlid,pro,anti,duration,pro_up,pro_down,anti_up,anti_down,pro_save,anti_save,start_time2, end_time2) %>%
  filter(str_detect(urlid,"mt_") & !is.na(pro))

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


w12 <- w12 %>%
  ungroup() %>%
  mutate(topic_id = str_replace(video_link_w2,".*&topicid=(.*)&allowDupe=1&id=(.*)$","\\1"),
         urlid = str_replace(video_link_w2,".*&topicid=(.*?)&allowDupe=1&id=(.*)$","\\2"),
  )

w12 <- left_join(w12,ytrecs,by=c("topic_id","urlid"))

w12 <- w12 %>%
  arrange(worker_id, start_time2) %>%
  group_by(worker_id) %>%
  slice(1) %>% # Keep first resp
  ungroup()
print("ISSUE 2 NUMBERS:")
print(paste('count w/ valid ytrecs data:', sum(!is.na(w12$pro))))
print(paste('count w/ valid ytrecs interactions:', sum(!is.na(w12$total_thumbs))))
print('interactions:')
summary(w12$total_interactions)


# create numeric dosage version of treatment:
w12 <- w12 %>%
  mutate(treatment_dose = dplyr::recode(treatment_arm,
                                 "anti_31"= 1, "anti_22" = 0,
                                 "pro_31"= 1, "pro_22" = 0,
                                 "control"=NA_real_),
         treatment_seed = str_replace(treatment_arm,"(.*)\\_\\d{2}","\\1")
  )

write_csv(w12, "../results/intermediate data/minimum wage (issue 2)/qualtrics_w12_clean.csv")
