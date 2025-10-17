cat(rep('=', 80),
    '\n\n',
    'OUTPUT FROM: minimum wage (issue 2)/01_trt_assign.R',
    '\n\n',
    sep = ''
    )
    
library(tidyverse)
library(janitor)
library(lubridate)
library(randomizr)
library(haven)

# create directory to hold cached intermediate files
dir.create("../results/intermediate data/minimum wage (issue 2)",
           recursive = TRUE, showWarnings = FALSE)


w1 <- read_csv("../data/minimum wage (issue 2)/YouTube+Min+Wage+-+Apr+2022+presurvey_May+24,+2022_02.57.csv")[-c(1,2),] %>% clean_names() %>%
  filter(finished == "True", q62 == "I agree to participate\u2028")

# Recodes ======================================================================

w1 <- w1 %>% mutate(start_date = as_datetime(start_date),
                    end_date = as_datetime(end_date),
                    survey_time = as.numeric(end_date-start_date))

print('wave 1 survey time:')
summary(w1$survey_time)

w1 <- w1 %>%
  mutate(man = ifelse(q26 == "Man", 1, 0),
         black = ifelse(str_detect(q29, "Black"), 1, 0),
         white = ifelse(str_detect(q29, "White"), 1, 0),
         college = ifelse(str_detect(q30, "college ") | str_detect(q30, "Post"), 1, 0),
         income_gt50k = ifelse(q31 %in% names(table(w1$q31))[c(2,3,5,10:13)], 1, 0)
  )

# PID:
w1$pid <- ifelse(w1$pid1=="Democrat",-1,NA)
w1$pid <- ifelse(w1$pid1=="Republican",1,w1$pid)
w1$pid <- ifelse(w1$pid4=="Closer to the Republican Party",1,w1$pid)
w1$pid <- ifelse(w1$pid4=="Closer to the Democratic Party",-1,w1$pid)
w1$pid <- ifelse(w1$pid4=="Neither",0,w1$pid)

print('wave 1 party id:')
round(table(w1$pid) / sum(table(w1$pid)), digits=2)

w1$ideo <- ifelse(w1$ideo1=="Liberal",-1,NA)
w1$ideo <- ifelse(w1$ideo1=="Conservative",1,w1$ideo)
w1$ideo <- ifelse(w1$ideo4=="Closer to liberals",-1,w1$ideo)
w1$ideo <- ifelse(w1$ideo4=="Closer to conservatives",1,w1$ideo)
w1$ideo <- ifelse(w1$ideo4=="Neither",0,w1$ideo)

print('wave 1 ideology:')
round(table(w1$ideo) / sum(table(w1$ideo)), digits=2)

w1$age <- 2022 - as.numeric(w1$q27)



# A/V check ====================================================================

print("audio check:")
length(which(w1$q87 == "Quick and easy")) / length(w1$q87)

print("video check:")
length(which(w1$q89 == "wikiHow")) / length(w1$q89)

w1$audio_ok <- 1*(w1$q87 == "Quick and easy")
w1$video_ok <- 1*(w1$q89 == "wikiHow")

# Convert pre-treatment DV to numeric unit scale -------------------------------

w1 <- w1 %>%
  mutate( # higher = more conservative
    minwage15 = recode(minwage15,"Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    rtwa_v1 = recode(rtwa_v1, "Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    #minwage_inflation = recode(minwage_inflation,"Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    rtwa_v2 = recode(rtwa_v2, "Strongly oppose"=4,"Somewhat oppose"=3,"Neither support nor oppose"=2,"Somewhat support"=1,"Strongly support"=0)/4,
    mw_support = recode(mw_support, "Strongly oppose raising the minimum wage"=4,"Somewhat oppose raising the minimum wage"=3,"Neither support nor oppose raising the minimum wage"=2,"Somewhat support raising the minimum wage"=1,"Strongly support raising the minimum wage"=0)/4,
    minwage_howhigh = recode(minwage_howhigh, "Much lower than the current level"=4,"Somewhat lower than the current level"=3,"About the current level"=2,"Somewhat higher than the current level"=1,"Much higher than the current level"=0)/4,
    mw_help = recode(mw_help, "Would hurt low-income workers\n10\n"=9,"9"=8,"8"=7,"7"=6,"6"=5,"5"=4,"4"=3,"3"=2,"2"=1,"Would help low-income workers\n1"=0)/9,
    mw_restrict = recode(mw_restrict, "Would restrict businesses' freedom\n1\n"=9,"2"=8,"3"=7,"4"=6,"5"=5,"6"=4,"7"=3,"8"=2,"9"=1,"Would protect workers from exploitation\n10\n"=0)/9,
    minwage_text = (25-as.numeric(minwage_text))/25
    )

w1 <- w1 %>%
  rowwise() %>%
  mutate(mw_index = mean(c(minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text),
                        na.rm=T))



# trim sample -------------------------------------------------------------

# We exclude respondents who took less than 120 seconds to complete the Wave 1 survey, failed either
# an audio check or a video check, as well as those whose gun policy opinions fall within the most
# extreme 5% of the gun policy index outcome (i.e. < 0.25 or > 4.75 on the 0-5 scale, to guard
# against eventual ceiling/floor effects; in our pilot study this was 15% of the sample).

w1 <- w1 %>% filter(audio_ok == 1, video_ok == 1)
w1 <- w1 %>% filter(survey_time >= 2)
w1 <- w1 %>% filter(mw_index >= 0.025, mw_index <= 0.975)
w1 <- w1 %>% filter(!is.na(worker_id))
w1 <- w1 %>% distinct(worker_id, .keep_all = TRUE)

print('mw index:')
summary(w1$mw_index)



# Block random assignment ======================================================

# We randomly assign respondents to both a seed video type (pro-gun vs. anti-gun) and a recommendation system (3/1 vs. 2/2)
# blocking on Wave 1 gun policy opinions. In the sample of respondents
# who will be invited to Wave 2, we form terciles of the Wave 1 gun policy opinion index, referring
# to the lower, middle and upper terciles as anti-gun, moderate and pro-gun respectively

w1$tercile <- cut(w1$mw_index, breaks = quantile(w1$mw_index, c(0, 1/3, 2/3, 1)), include.lowest = TRUE, labels = 1:3)
tapply(w1$mw_index, w1$tercile, mean)
table(w1$tercile)

# pure control (with 1/5 probability), anti-MW 2/2 (with 2/5 probability), or anti-MW 3/1 (with 2/5 probability).
# seed position (pro-MW or anti-MW), recommendation system (2/2 or 3/1), or a
# pure control group (i.e. one of five possible conditions) with equal probability

# For MTurk --------------------------------------------------------------------

set.seed(2022)

w1$trt_system <- block_ra(blocks = w1$tercile, prob_each = c(2/5, 2/5, 1/5), conditions = c("2/2", "3/1", "pure control"))

w1$seed <- rep("", nrow(w1))
w1[w1$tercile == 1,]$seed <- "pro-minwage seed"
w1[w1$tercile == 3,]$seed <- "anti-minwage seed"
w1[w1$tercile == 2,]$seed <- complete_ra(N = length(which(w1$tercile == 2)), prob = 0.5, conditions = c("pro-minwage seed",
                                                                                                        "anti-minwage seed"))
with(w1[w1$tercile == 1,], round(prop.table(table(seed, trt_system)), digits = 3))
with(w1[w1$tercile == 2,], round(prop.table(table(seed, trt_system)), digits = 3))
with(w1[w1$tercile == 3,], round(prop.table(table(seed, trt_system)), digits = 3))

w1 <- w1 %>% mutate(trt_assign = case_when(seed == "anti-minwage seed" & trt_system == "2/2" ~ 1,
                                           seed == "anti-minwage seed" & trt_system == "3/1" ~ 2,
                                           seed == "pro-minwage seed" & trt_system == "2/2" ~ 3,
                                           seed == "pro-minwage seed" & trt_system == "3/1" ~ 4,
                                           trt_system == "pure control" ~ 5))

print('treatment assignment:')
table(w1$trt_assign)
print('seed assignment:')
table(w1$seed)
print('system assignment:')
table(w1$trt_system)
print('seed & system assignment:')
table(w1$trt_system, w1$seed)

# w1$batch <- sample(c(rep(1:floor(nrow(w1)/500), 500), rep(6, nrow(w1)-500*5)))
# sent to Qualtrics
# write_csv(data.frame(trt = w1$trt_assign, id = w1$worker_id), "mw_mturk_wave1_assignments.csv")



# YouGov -----------------------------------------------------------------------

w1 <- read_sav("../data/minimum wage (issue 2)/PRIN0016_W1_OUTPUT.sav") %>% filter(consent == 22)
w1$caseid <- as.character(w1$caseid)

# Convert pre-treatment DV to numeric unit scale
w1 <- w1 %>%
  mutate( # higher = more conservative
    minwage15 = (minwage15-1)/4,
    rtwa_v1 = (RTWA_v1-1)/4,
    rtwa_v2 = (RTWA_v2-1)/4,
    mw_support = (mw_support-1)/4,
    minwage_howhigh = (minwage_howhigh-1)/4,
    mw_help = (mw_help_a-1)/9,
    mw_restrict = (10-mw_restrict_1)/9,
    minwage_text = (25-minwage_text)/25
  )


w1 <- w1 %>%
  rowwise() %>%
  mutate(mw_index = mean(c(minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help, mw_restrict, minwage_text),
                         na.rm=T))

w1 <- w1 %>% mutate(start_date = as_datetime(starttime),
                    end_date = as_datetime(endtime),
                    survey_time = as.numeric(end_date-start_date))

print('wave 1 survey time:')
summary(w1$survey_time)

w1 <- w1 %>% filter(survey_time >= 2)
w1 <- w1 %>% filter(mw_index >= 0.025, mw_index <= 0.975)

print('mw index:')
summary(w1$mw_index)

w1$tercile <- cut(w1$mw_index, breaks = quantile(w1$mw_index, c(0, 1/3, 2/3, 1)), include.lowest = TRUE, labels = 1:3)

write_csv(select(w1, caseid, tercile, mw_index), "../results/intermediate data/minimum wage (issue 2)/yougov_terciles.csv")


# pure control (with 1/5 probability), anti-MW 2/2 (with 2/5 probability), or anti-MW 3/1 (with 2/5 probability).
# seed position (pro-MW or anti-MW), recommendation system (2/2 or 3/1), or a
# pure control group (i.e. one of five possible conditions) with equal probability

set.seed(22022)

# For YouGov
w1$trt_system <- block_ra(blocks = w1$tercile, prob_each = c(.5, .5), conditions = c("2/2", "3/1"))

w1$seed <- rep("", nrow(w1))
w1[w1$tercile == 1,]$seed <- "pro-minwage seed"
w1[w1$tercile == 3,]$seed <- "anti-minwage seed"
w1[w1$tercile == 2,]$seed <- complete_ra(N = length(which(w1$tercile == 2)), prob = 0.5, conditions = c("pro-minwage seed",
                                                                                                        "anti-minwage seed"))
with(w1[w1$tercile == 1,], round(prop.table(table(seed, trt_system)), digits = 3))
with(w1[w1$tercile == 2,], round(prop.table(table(seed, trt_system)), digits = 3))
with(w1[w1$tercile == 3,], round(prop.table(table(seed, trt_system)), digits = 3))

w1 <- w1 %>% mutate(trt_assign = case_when(seed == "anti-minwage seed" & trt_system == "2/2" ~ 1,
                                           seed == "anti-minwage seed" & trt_system == "3/1" ~ 2,
                                           seed == "pro-minwage seed" & trt_system == "2/2" ~ 3,
                                           seed == "pro-minwage seed" & trt_system == "3/1" ~ 4))

print('treatment assignment:')
table(w1$trt_assign)
print('seed assignment:')
table(w1$seed)
print('system assignment:')
table(w1$trt_system)
print('seed & system assignment:')
table(w1$trt_system, w1$seed)

# sent to YouGov
# write_csv(select(w1, caseid, trt_system, seed, trt_assign), "mw_yg_wave1_assignments.csv")
