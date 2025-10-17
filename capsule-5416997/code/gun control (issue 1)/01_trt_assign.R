cat(rep('=', 80),
    '\n\n',
    'OUTPUT FROM: gun control (issue 1)/01_trt_assign.R',
    '\n\n',
    sep = ''
    )
    
library(tidyverse)
library(janitor)
library(lubridate)
library(randomizr)

# create directory to hold cached intermediate files
dir.create("../results/intermediate data/gun control (issue 1)/",
           recursive = TRUE, showWarnings = FALSE)

w1 <- read_csv("../data/gun control (issue 1)/wave1_final.csv")[-c(1,2),] %>%
  clean_names() %>%
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
w1$pid[w1$pid4=="Closer to the Republican Party"] <- 1
w1$pid[w1$pid4=="Closer to the Democratic Party"] <- -1
w1$pid[w1$pid4=='Neither'] <- 0

print('wave 1 party id:')
round(table(w1$pid) / sum(table(w1$pid)), digits=2)

w1$ideo <- ifelse(w1$ideo1=="Liberal",-1,NA)
w1$ideo <- ifelse(w1$ideo1=="Conservative",1,w1$ideo)
w1$ideo[w1$ideo4=="Closer to liberals"] <- -1
w1$ideo[w1$ideo4=="Closer to conservatives"] <- 1
w1$ideo[w1$ideo4=="Neither"] <- 0

print('wave 1 ideology:')
round(table(w1$ideo) / sum(table(w1$ideo)), digits=2)

w1$age <- 2021 - as.numeric(w1$q27)



# A/V check ====================================================================

print("audio check:")
length(which(w1$q87 == "Quick and easy")) / length(w1$q87)

print("video check:")
length(which(w1$q89 == "wikiHow")) / length(w1$q89)

w1$audio_ok <- 1*(w1$q87 == "Quick and easy")
w1$video_ok <- 1*(w1$q89 == "wikiHow")

w1 <- w1 %>%
  mutate(gun_own = dplyr::recode(q15, "Yes" = 1, "No" = 0))

# Convert pre-treatment DV to numeric unit scale -------------------------------

w1 <- w1 %>%
  mutate( # higher = more pro-gun
    right_to_own_importance = recode(q79, "Protect the right to own guns" = 1, "Regulate gun ownership" = 0),
    assault_ban = (match(q81, names(table(q81))[c(5,3,1,2,4)])-1)/4,
    handgun_ban = (match(q82, names(table(q82))[c(5,3,1,2,4)])-1)/4,
    concealed_safe = 1-(match(q83, names(table(q83))[c(2,5,3,4,1)])-1)/4,
    stricter_laws = (match(q23, names(table(q23))[c(5,3,1,2,4)])-1)/4
  )

w1 <- w1 %>%
  rowwise() %>%
  mutate(gun_index = sum(c(right_to_own_importance,assault_ban,handgun_ban,concealed_safe,stricter_laws), na.rm=T),
         gun_index_2 = mean(c(right_to_own_importance,assault_ban,handgun_ban,concealed_safe), na.rm=T))

# Cronbach's alpha -------------------------------------------------------------

index_fa <- psych::alpha(select(w1, right_to_own_importance, assault_ban, handgun_ban, concealed_safe, stricter_laws), check.keys = TRUE)
alpha <- index_fa$total["raw_alpha"]
writeLines(as.character(round(alpha,2)),
           con = "../results/guncontrol_outcomes_alpha.tex",sep = "%")



# trim sample -------------------------------------------------------------

# We exclude respondents who took less than 120 seconds to complete the Wave 1 survey, failed either
# an audio check or a video check, as well as those whose gun policy opinions fall within the most
# extreme 5% of the gun policy index outcome (i.e. < 0.25 or > 4.75 on the 0-5 scale, to guard
# against eventual ceiling/floor effects; in a pilot study this was 15% of the sample).

w1 <- w1 %>% filter(audio_ok == 1, video_ok == 1)
w1 <- w1 %>% filter(survey_time >= 2)
w1 <- w1 %>% filter(gun_index >= 0.25, gun_index <= 4.75)

print('gun index:')
summary(w1$gun_index)



# Block random assignment ======================================================

# We randomly assign respondents to both a seed video type (pro-gun vs. anti-gun) and a recommendation system (3/1 vs. 2/2)
# blocking on Wave 1 gun policy opinions. In the sample of respondents
# who will be invited to Wave 2, we form terciles of the Wave 1 gun policy opinion index, referring
# to the lower, middle and upper terciles as anti-gun, moderate and pro-gun respectively

w1$tercile <- cut(w1$gun_index, breaks = quantile(w1$gun_index, c(0, 1/3, 2/3, 1)), include.lowest = TRUE, labels = 1:3)
tapply(w1$gun_index, w1$tercile, mean)
table(w1$tercile)

#  pure control (with 1/5 probability), anti-gun 2/2 (with 2/5 probability), or anti-gun 3/1 (with 2/5 probability).
# seed position (pro-gun or anti-gun), recommendation system (2/2 or 3/1), or a
# pure control group (i.e. one of five possible conditions) with equal probability

set.seed(2021)

w1$trt_system <- block_ra(blocks = w1$tercile, prob_each = c(2/5, 2/5, 1/5), conditions = c("2/2", "3/1", "pure control"))

w1$seed <- rep("", nrow(w1))
w1[w1$tercile == 1,]$seed <- "anti-gun seed"
w1[w1$tercile == 3,]$seed <- "pro-gun seed"
w1[w1$tercile == 2,]$seed <- complete_ra(N = length(which(w1$tercile == 2)), prob = 0.5, conditions = c("pro-gun seed",
                                                                                                        "anti-gun seed"))
with(w1[w1$tercile == 1,], round(prop.table(table(seed, trt_system)), digits = 3))
with(w1[w1$tercile == 2,], round(prop.table(table(seed, trt_system)), digits = 3))
with(w1[w1$tercile == 3,], round(prop.table(table(seed, trt_system)), digits = 3))

w1 <- w1 %>% mutate(trt_assign = case_when(seed == "anti-gun seed" & trt_system == "2/2" ~ 1,
                                           seed == "anti-gun seed" & trt_system == "3/1" ~ 2,
                                           seed == "pro-gun seed" & trt_system == "2/2" ~ 3,
                                           seed == "pro-gun seed" & trt_system == "3/1" ~ 4,
                                           trt_system == "pure control" ~ 5))

print('treatment assignment:')
table(w1$trt_assign)
print('seed assignment:')
table(w1$seed)
print('system assignment:')
table(w1$trt_system)
print('seed & system assignment:')
table(w1$trt_system, w1$seed)

w1$batch <- sample(c(rep(1:floor(nrow(w1)/500), 500), rep(6, nrow(w1)-500*5)))

# sent to Qualtrics ------------------------------------------------------------

# write_csv(data.frame(trt = w1$trt_assign, id = w1$worker_id, batch = w1$batch),
#           "guncontrol_wave1_assignments.csv")
