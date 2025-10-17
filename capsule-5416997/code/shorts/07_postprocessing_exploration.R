cat(rep('=', 80),
    '\n\n',
    'OUTPUT FROM: shorts/07_postprocessing_exploration.R',
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
library(psych)

w12 <- read_csv("../results/intermediate data/shorts/qualtrics_w12_clean_ytrecs_may2024.csv")

## SAMPLE SIZE AND CRONBACH'S ALPHA ------------------

# SAMPLE SIZE
w12 %>%
  filter(!is.na(treatment_arm)) %>%
  count() %>%
  as.integer() %>%
  format(big.mark = ',')

# CRONBACH'S ALPHA ON POLICY INDEX
w12 %>%
  select(minwage15_pre,
         rtwa_v1_pre,
         rtwa_v2_pre,
         mw_support_pre,
         minwage_howhigh_pre,
         mw_help_pre_1,
         mw_restrict_pre_1,
         minwage_text_r_pre
  ) %>%
  alpha() %>%
  `[[`('total') %>%
  `[`('raw_alpha') %>%
  as.numeric() %>%
  format(digits = 2, nsmall = 2) %>%
  paste0('%') %>%  # trailing comment char to prevent latex import issue
  writeLines('../results/alpha_study4.txt')


# FACTOR ANALYSIS WITH VARIMAX ROTATION (PRE)
pca2 <- psych::principal(select(w12, minwage15_pre, rtwa_v1_pre, 
                                rtwa_v2_pre, mw_support_pre, minwage_howhigh_pre, 
                                mw_help_pre_1, mw_restrict_pre_1, minwage_text_r_pre),
                         rotate="varimax",
                         nfactors=1
)
pc2 <- pca2$Vaccounted[2]
writeLines(as.character(round(pc2, 2)),con = "../results/outcomes_pc2_study4_pre.tex",sep = "%")


# FACTOR ANALYSIS WITH VARIMAX ROTATION (POST)
pca2 <- psych::principal(
  select(w12, minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, mw_help_1, 
         mw_restrict_1, minwage_text_r),
  rotate="varimax",
  nfactors=1
)
pc2 <- pca2$Vaccounted[2]
writeLines(as.character(round(pc2, 2)),con = "../results/outcomes_pc2_study4_post.tex",sep = "%")


## BASIC DESCRIPTIVE FIGURES ------------------

## TIME SPENT DURING THE SURVEY
(surveytime_plot <- ggplot(w12) +
   geom_histogram(aes(x=survey_time,y=..density../sum(..density..))) +
   scale_x_continuous("Overall survey time taken (minutes)",
                       breaks=seq(0,100,10),
                       limits=c(-1,100)
                      ) +
   scale_y_continuous("Density") +
   geom_vline(xintercept = mean(w12$survey_time,na.rm=T),lty=3,col="red") +
   annotate(x=mean(w12$survey_time+1,na.rm=T),y=0.13,geom = "text",
            label=paste0("Average: ",round(mean(w12$survey_time,na.rm=T),0)," minutes"),hjust=0) +
   geom_vline(xintercept = median(w12$survey_time,na.rm=T),lty=2,col="red") +
   annotate(x=median(w12$survey_time+1,na.rm=T),y=0.16,geom = "text",
            label=paste0("Median: ",round(median(w12$survey_time,na.rm=T),0)," minutes"),hjust=0) +
   theme_minimal()
)

## TIME SPENT ON THE INTERFACE
(ytrecstime_plot <- ggplot(w12) +
    geom_histogram(aes(x=interface_duration/60,y=..density../sum(..density..))) +
    scale_x_continuous("Interface Time Taken (minutes)",
                       breaks=seq(0,80,10),
                       limits=c(-1,70)) +
    scale_y_continuous("Density") +
    geom_vline(xintercept = mean(w12$interface_duration/60,na.rm=T),lty=3,col="red") +
    annotate(x=mean(w12$interface_duration/60+1,na.rm=T),y=0.1,geom = "text",
             label=paste0("Average: ",round(mean(w12$interface_duration/60,na.rm=T),0)," minutes"),hjust=0) +
    geom_vline(xintercept = median(w12$interface_duration/60,na.rm=T),lty=2,col="red") +
    annotate(x=median(w12$interface_duration/60+1,na.rm=T),y=0.13,geom = "text",
             label=paste0("Median: ",round(median(w12$interface_duration/60,na.rm=T),0)," minutes"),hjust=0) +
    theme_minimal()
)

## PRE OPINIONS OVERALL
(hist_mwindex <- ggplot(w12) +
    geom_histogram(aes(x=mw_index_pre)) +
    scale_x_continuous("Minimum Wage Opinions Index, Pre") +
    scale_y_continuous("Count",limits=c(-5,200)) +
    annotate(x = 0.92,y=-3,geom = "text",label="More conservative\nopinions",col="red",hjust=1,size=3,lineheight=0.75) +
    annotate(x = 0.98,xend=1,y=-3,yend=-3,geom = "segment",arrow=arrow(type = "closed",angle = 20),col="red") +
    annotate(x = 0.08,y=-3,geom = "text",label="More liberal\nopinions",col="blue",hjust=0,size=3,lineheight=0.75) +
    annotate(x = 0.02,xend=0.00,y=-3,yend=-3,geom = "segment",arrow=arrow(type = "closed",angle = 20),col="blue") +
    theme_minimal()
)

## PRE OPINION BY TERCILE
(hist_mwindex_thirds <- ggplot(w12,aes(x=mw_index_pre)) +
    geom_histogram(data=filter(w12,thirds==1),aes(x=mw_index_pre),fill="blue") +
    geom_histogram(data=filter(w12,thirds==2),aes(x=mw_index_pre),fill="grey") +
    geom_histogram(data=filter(w12,thirds==3),aes(x=mw_index_pre),fill="red") +
    scale_x_continuous("Minimum Wage Opinions Index, Pre") +
    scale_y_continuous("Count",limits=c(-5,200)) +
    annotate(x = 0.92,y=-5,geom = "text",label="More conservative\nopinions",col="red",hjust=1,size=3,lineheight=0.75) +
    annotate(x = 0.98,xend=1,y=-5,yend=-5,geom = "segment",arrow=arrow(type = "closed",angle = 20),col="red") +
    annotate(x = 0.08,y=-5,geom = "text",label="More liberal\nopinions",col="blue",hjust=0,size=3,lineheight=0.75) +
    annotate(x = 0.02,xend=0.00,y=-5,yend=-5,geom = "segment",arrow=arrow(type = "closed",angle = 20),col="blue") +
    theme_minimal()
)

(hist_mwindex_thirds_nocolor <- ggplot(w12,aes(x=mw_index_pre)) +
    geom_histogram(data=filter(w12,thirds==1),aes(x=mw_index_pre),fill="grey") +
    geom_histogram(data=filter(w12,thirds==2),aes(x=mw_index_pre),fill="grey") +
    geom_histogram(data=filter(w12,thirds==3),aes(x=mw_index_pre),fill="grey") +
    scale_x_continuous("Minimum Wage Opinions Index, W1") +
    scale_y_continuous("Count",limits=c(-5,200)) +
    annotate(x = 0.92,y=-5,geom = "text",label="More conservative\nopinions",col="red",hjust=1,size=3,lineheight=0.75) +
    annotate(x = 0.98,xend=1,y=-5,yend=-5,geom = "segment",arrow=arrow(type = "closed",angle = 20),col="red") +
    annotate(x = 0.08,y=-5,geom = "text",label="More liberal\nopinions",col="blue",hjust=0,size=3,lineheight=0.75) +
    annotate(x = 0.02,xend=0.00,y=-5,yend=-5,geom = "segment",arrow=arrow(type = "closed",angle = 20),col="blue") +
    theme_minimal()
)

# SUMMARY PRE OPINIONS FOR EACH CONDITION
groupsumm_bythirds <- w12 %>%
  group_by(treatment_arm,thirds) %>%
  summarize(n = n()) %>%
  na.omit() %>%
  mutate(treatment_arm = factor(treatment_arm,levels=c("pc", "pi","ac" , "ai"),
                                labels = c("Liberal\nconstant",
                                           "Liberal\nincreasing",
                                           "Conservative\nconstant",
                                           "Conservative\nincreasing"),ordered=T),
         thirds = factor(thirds,levels=c(1,2,3),ordered=T))

groupsumm <- w12 %>%
  group_by(treatment_arm) %>%
  summarize(
    minwage15 = mean(minwage15_pre,na.rm=T),
    rtwa_v1 =  mean(rtwa_v1_pre, na.rm = T),
    rtwa_v2 = mean(rtwa_v2_pre, na.rm = T),
    mw_support = mean(mw_support_pre,na.rm = T),
    minwage_howhigh = mean(minwage_howhigh_pre, na.rm = T),
    mw_help_1 = mean(mw_help_pre_1, na.rm = T),
    mw_restrict_1 = mean(mw_restrict_pre_1,na.rm = T),
    minwage_text_r = mean(minwage_text_r_pre,na.rm = T),
    mw_index_pre = mean(mw_index_pre,na.rm = T),
    n = n()) %>%
  na.omit() %>%
  mutate(treatment_arm = factor(treatment_arm,levels=c("pc", 
                                                       "pi",
                                                       "ac" , 
                                                       "ai"),
                                labels = c("Liberal\nconstant",
                                           "Liberal\nincreasing",
                                           "Conservative\nconstant",
                                           "Conservative\nincreasing"),ordered=T))

# N IN EACH TREATMENT CONDITION
(plot_hist_n <- ggplot(groupsumm) +
    geom_bar(aes(x=treatment_arm,y=n),stat="identity") +
    geom_text(aes(x=treatment_arm,y=n+15,label=n),stat="identity") +
    scale_x_discrete("Treatment Condition") +
    scale_y_continuous("N") +
    theme_minimal()
)

## N IN EACH TREATMENT CONDITION COLORED BY THIRDS
(plot_hist_n_bythirds <- ggplot(groupsumm_bythirds) +
    geom_bar(aes(x=treatment_arm,y=n,fill=thirds),stat="identity") +
    geom_text(data=groupsumm,aes(x=treatment_arm,y=n+15,label=n),stat="identity") +
    scale_x_discrete("Treatment Condition") +
    scale_y_continuous("N") +
    scale_fill_manual("Tercile of\nPre-Opinion",breaks=c(1,2,3),values=c("blue","grey","red")) +
    theme_minimal()
)

## AVERAGE PRE-OPINION ON MINIMUM WAGE INDEX
(plot_hist_mwindex <- ggplot(groupsumm) +
    geom_bar(aes(x=treatment_arm,y=mw_index_pre),stat="identity") +
    scale_x_discrete("Treatment Condition") +
    scale_y_continuous("Average Pre-Opinion\non Minimum Wage Index",
                       limits=c(0,0.6),
                       breaks = seq(0,0.6,0.2),
                       labels=c("\n0.0\nMore\nliberal\nopinions","0.2","0.4","More\nconservative\nopinions\n0.6\n\n\n")) +
    theme_minimal() +
    theme(plot.margin = unit(c(1.75,0.5,0.5,0.5),"lines"))
)

# SUMMARY FOR EACH CONDITION
groupsumm <- w12 %>%
  group_by(treatment_arm) %>%
  summarize(
    minwage15 = mean(minwage15,na.rm=T),
    rtwa_v1 =  mean(rtwa_v1, na.rm = T),
    rtwa_v2 = mean(rtwa_v2, na.rm = T),
    mw_support = mean(mw_support,na.rm = T),
    minwage_howhigh = mean(minwage_howhigh, na.rm = T),
    mw_help_1 = mean(mw_help_1, na.rm = T),
    mw_restrict_1 = mean(mw_restrict_1,na.rm = T),
    minwage_text_r = mean(minwage_text_r,na.rm = T),
    mw_index = mean(mw_index,na.rm = T),
    n = n()) %>%
  na.omit() %>%
  mutate(treatment_arm = factor(treatment_arm,levels=c("pc", 
                                                       "pi",
                                                       "ac" , 
                                                       "ai"),
                                labels = c("Liberal\nconstant",
                                           "Liberal\nincreasing",
                                           "Conservative\nconstant",
                                           "Conservative\nincreasing"),
                                ordered=T))

(plot_hist_mwindex <- ggplot(groupsumm) +
    geom_bar(aes(x=treatment_arm,y=mw_index),stat="identity") +
    scale_x_discrete("Treatment Condition") +
    scale_y_continuous("Average Post-Opinion\non Minimum Wage Index",
                       limits=c(0,0.6),
                       breaks = seq(0,0.6,0.2),
                       labels=c("\n0.0\nMore\nliberal\nopinions","0.2","0.4","More\nconservative\nopinions\n0.6\n\n\n")) +
    theme_minimal() +
    theme(plot.margin = unit(c(1.75,0.5,0.5,0.5),"lines"))
)

## CHANGES IN OPINION BETWEEN WAVES
treatsumm <- w12 %>%
  group_by(treatment_arm) %>%
  summarize(minwage15 = mean(minwage15-minwage15_pre,na.rm=T),
            rtwa_v1 =  mean(rtwa_v1-rtwa_v1_pre, na.rm = T),
            rtwa_v2 = mean(rtwa_v2-rtwa_v2_pre, na.rm = T),
            mw_support = mean(mw_support-mw_support_pre,na.rm = T),
            minwage_howhigh = mean(minwage_howhigh-minwage_howhigh_pre, na.rm = T),
            mw_help_1 = mean(mw_help_1-mw_help_pre_1, na.rm = T),
            mw_restrict_1 = mean(mw_restrict_1-mw_restrict_pre_1,na.rm = T),
            minwage_text_r = mean(minwage_text_r-minwage_text_r_pre,na.rm = T),
            mw_index_change = mean(mw_index - mw_index_pre,na.rm = T),
            n = n()) %>%
  na.omit() %>%
  mutate(treatment_arm = factor(treatment_arm,levels=c("pc", 
                                                       "pi",
                                                       "ac" , 
                                                       "ai"),
                                labels = c("Liberal\nconstant",
                                           "Liberal\nincreasing",
                                           "Conservative\nconstant",
                                           "Conservative\nincreasing"),
                                ordered=T))

w1w2_corrplot <- corrplot::corrplot(cor(select(w12,
                                               minwage15_pre, rtwa_v1_pre, rtwa_v2_pre, mw_support_pre, 
                                               minwage_howhigh_pre, mw_help_pre_1, mw_restrict_pre_1, minwage_text_r_pre,
                                               minwage15, rtwa_v1, rtwa_v2, mw_support, minwage_howhigh, 
                                               mw_help_1, mw_restrict_1, minwage_text_r), use = "complete.obs")[1:8,9:16],method = "shade")
dev.off()

## AVERAGE OPINION CHANGE POST-PRE ON MIN WAGE POLICY INDEX
(plot_hist_mwindex <- ggplot(treatsumm) +
    geom_bar(aes(x=treatment_arm,y=mw_index_change),stat="identity") +
    scale_x_discrete("Treatment Condition") +
    scale_y_continuous("Average Opinion Change Post-Pre\non Min. Wage Policy Index",
                       limits=c(-0.2,0.2),
                       breaks = seq(-0.2,0.2,0.1),
                       labels=c("\n\n\n-0.2\nLiberal\nopinion\nchange","-0.1","0.00","0.1","Conservative\nopinion\nchange\n0.2\n\n\n")
                       ) +
    theme_minimal() +
    theme(plot.margin = unit(c(1.75,0.5,0.5,0.5),"lines"))
)

### CHANGE FOR MODERATES
treatsumm_thirds <- w12 %>%
  group_by(thirds, treatment_arm) %>%
  summarize(minwage15 = mean(minwage15-minwage15_pre,na.rm=T),
            rtwa_v1 =  mean(rtwa_v1-rtwa_v1_pre, na.rm = T),
            rtwa_v2 = mean(rtwa_v2-rtwa_v2_pre, na.rm = T),
            mw_support = mean(mw_support-mw_support_pre,na.rm = T),
            minwage_howhigh = mean(minwage_howhigh-minwage_howhigh_pre, na.rm = T),
            mw_help_1 = mean(mw_help_1-mw_help_pre_1, na.rm = T),
            mw_restrict_1 = mean(mw_restrict_1-mw_restrict_pre_1,na.rm = T),
            minwage_text_r = mean(minwage_text_r-minwage_text_r_pre,na.rm = T),
            mw_index_change = mean(mw_index - mw_index_pre,na.rm = T),
            n = n()) %>%
  na.omit() %>%
  mutate(treatment_arm = factor(treatment_arm,levels=c("pc", 
                                                       "pi",
                                                       "ac" , 
                                                       "ai"),
                                labels = c("Liberal\nconstant",
                                           "Liberal\nincreasing",
                                           "Conservative\nconstant",
                                           "Conservative\nincreasing"),
                                ordered=T))

(plot_hist_mwindex_thirds <- ggplot(treatsumm_thirds %>% filter(thirds == 2)) +
    geom_bar(aes(x=treatment_arm,y=mw_index_change),stat="identity") +
    scale_x_discrete("Treatment Condition") +
    scale_y_continuous("Average Opinion Change Post-Pre\non Min. Wage Policy Index\nfor Moderates",
                       limits=c(-0.2,0.2),
                       breaks = seq(-0.2,0.2,0.1),
                       labels=c("\n\n\n-0.2\nLiberal\nopinion\nchange","-0.1","0.00","0.1","Conservative\nopinion\nchange\n0.2\n\n\n")
    ) +
    theme_minimal() +
    theme(plot.margin = unit(c(1.75,0.5,0.5,0.5),"lines"))
)


## BASE CONTROL FIGURES --------------------------------------

##
## RUN 04_analysis_multipletesting_basecontrol_may2024.R, THEN READ IN ADJUSTED P-VALUES
##

coefs_basecontrol <- read_csv("../results/padj_basecontrol_pretty_ytrecs_may2024.csv")

outcome_labels <- data.frame(outcome = c("Minimum wage\nindex"),
                             specificoutcome = c("mw_index"),
                             family = c(rep("Policy Attitudes\n(unit scale, + is more conservative)",1)))


#### THE effect of INCREASING vs. CONSTANT assignment among LIBERAL participants ####
coefs_third1_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.pro:recsys.pi - attitude.pro:recsys.pc" &
           layer3_specificoutcome != "overall")


coefs_third1_basecontrol$outcome = outcome_labels$outcome[match(coefs_third1_basecontrol$layer3_specificoutcome,
                                                                outcome_labels$specificoutcome)]

coefs_third1_basecontrol$family = outcome_labels$family[match(coefs_third1_basecontrol$layer3_specificoutcome,outcome_labels$specificoutcome)]

coefs_third1_basecontrol <- mutate(coefs_third1_basecontrol,
                                   family = factor(family,
                                                   levels = c("Policy Attitudes\n(unit scale, + is more conservative)"
                                                     ),ordered = T))

coefs_third1_basecontrol <- coefs_third1_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_third1_basecontrol):1
  )

writeLines(as.character(round(100*abs(filter(coefs_third1_basecontrol,layer3_specificoutcome=="pro_fraction_chosen")$est),0)),
           con = "../results/beta_recsys_pro_fraction_chosen_third1.tex",sep="%")


#### THE effect of INCREASING vs. CONSTANT assignment among LIBERAL participants ####
(coefplot_third1_basecontrol <- ggplot(filter(coefs_third1_basecontrol),aes(y=plotorder)) +
    geom_errorbarh(aes(xmin=ci_lo_95,xmax=ci_hi_95),height=0,lwd=0.5) +
    geom_errorbarh(aes(xmin=ci_lo_90,xmax=ci_hi_90),height=0,lwd=1) +
    geom_point(aes(x=est),size=1.5) +
    geom_vline(xintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_y_continuous("",
                       breaks = coefs_third1_basecontrol$plotorder,
                       labels = coefs_third1_basecontrol$outcome) +
    scale_x_continuous("Increasing Liberal seed vs. Constant Liberal seed assignment \namong Liberal participants \n(95% and 90% CIs)") +
    coord_cartesian(xlim=c(-0.2,0.2)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"))
)

#### THE effect of INCREASING vs. CONSTANT assignment among CONSERVATIVE participants ####
coefs_third3_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.anti:recsys.ai - attitude.anti:recsys.ac" &
           layer3_specificoutcome != "overall")

coefs_third3_basecontrol$outcome = outcome_labels$outcome[match(coefs_third3_basecontrol$layer3_specificoutcome,
                                                                outcome_labels$specificoutcome)]

coefs_third3_basecontrol$family = outcome_labels$family[match(coefs_third3_basecontrol$layer3_specificoutcome,
                                                              outcome_labels$specificoutcome)]

coefs_third3_basecontrol <- mutate(coefs_third3_basecontrol,
                                   family = factor(family,levels = c("Policy Attitudes\n(unit scale, + is more conservative)"
                                                                     ),ordered = T))


coefs_third3_basecontrol <- coefs_third3_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_third3_basecontrol):1
  )

writeLines(as.character(round(100*abs(filter(coefs_third3_basecontrol,layer3_specificoutcome=="pro_fraction_chosen")$est),0)),con = "../results/beta_recsys_pro_fraction_chosen_third3.tex",sep="%")
writeLines(as.character(round(abs(filter(coefs_third3_basecontrol,layer3_specificoutcome=="mw_index_w2")$est),2)),con = "../results/beta_recsys_mwindex_third3.tex",sep="%")
writeLines(as.character(round(abs(filter(coefs_third3_basecontrol,layer3_specificoutcome=="mw_index_w2")$ci_hi_95),2)),con = "../results/cihi_recsys_mwindex_third3.tex",sep="%")


#### THE effect of INCREASING vs. CONSTANT assignment among CONSERVATIVE participants ####
(coefplot_third3_basecontrol <- ggplot(filter(coefs_third3_basecontrol),aes(y=plotorder)) +
    geom_errorbarh(aes(xmin=ci_lo_95,xmax=ci_hi_95),height=0,lwd=0.5) +
    geom_errorbarh(aes(xmin=ci_lo_90,xmax=ci_hi_90),height=0,lwd=1) +
    geom_point(aes(x=est),size=1.5) +
    geom_vline(xintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_y_continuous("",
                       breaks = coefs_third3_basecontrol$plotorder,labels = coefs_third3_basecontrol$outcome) +
    scale_x_continuous("Increasing Conservative vs. Constant Conservative \n seed among Conservative participants \n(95% and 90% CIs)") +
    coord_cartesian(xlim=c(-0.2,0.2)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"))
)

#### THE effect of INCREASING vs. CONSTANT assignment among MODERATE participants assigned to a LIBERAL sequence ####
coefs_third2_pro_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.neutral:recsys.pi - attitude.neutral:recsys.pc" &
         layer3_specificoutcome != "overall")


coefs_third2_pro_basecontrol$outcome = outcome_labels$outcome[match(coefs_third2_pro_basecontrol$layer3_specificoutcome,
                                                                    outcome_labels$specificoutcome)]

coefs_third2_pro_basecontrol$family = outcome_labels$family[match(coefs_third2_pro_basecontrol$layer3_specificoutcome,
                                                                  outcome_labels$specificoutcome)]

coefs_third2_pro_basecontrol <- mutate(coefs_third2_pro_basecontrol,
                                       family = factor(family,levels = c("Policy Attitudes\n(unit scale, + is more conservative)"
                                                                         ),ordered = T))

coefs_third2_pro_basecontrol <- coefs_third2_pro_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_third2_pro_basecontrol):1
  )
writeLines(as.character(round(100*abs(filter(coefs_third2_pro_basecontrol,layer3_specificoutcome=="pro_fraction_chosen")$est),0)),con = "../results/beta_recsys_pro_fraction_chosen_third2_proseed.tex",sep="%")
writeLines(as.character(abs(round(filter(coefs_third2_pro_basecontrol,layer3_specificoutcome=="platform_duration")$est,2))),con = "../results/beta_recsys_duration_third2_proseed.tex",sep="%")
writeLines(as.character(abs(round(filter(coefs_third2_pro_basecontrol,layer3_specificoutcome=="platform_duration")$est*60,1))),con = "../results/beta_minutes_recsys_duration_third2_proseed.tex",sep="%")

#### THE effect of INCREASING vs. CONSTANT assignment among MODERATE participants assigned to a LIBERAL sequence ####
(coefplot_third2_pro_basecontrol <- ggplot(filter(coefs_third2_pro_basecontrol),aes(y=plotorder)) +
    geom_errorbarh(aes(xmin=ci_lo_95,xmax=ci_hi_95),height=0,lwd=0.5) +
    geom_errorbarh(aes(xmin=ci_lo_90,xmax=ci_hi_90),height=0,lwd=1) +
    geom_point(aes(x=est),size=1.5) +
    geom_vline(xintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_y_continuous("",
                       breaks = coefs_third2_pro_basecontrol$plotorder,labels = coefs_third2_pro_basecontrol$outcome) +
    scale_x_continuous("Increasing Liberal vs. Constant Liberal seed among Moderates \n(95% and 90% CIs)") +
    coord_cartesian(xlim=c(-0.2,0.2)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"))
)
ggsave(coefplot_third2_pro_basecontrol,
       filename = "../results/coefplot_third2_pro_basecontrol.png",width=5,height=8)

#### THE effect of INCREASING vs. CONSTANT assignment among MODERATE participants assigned to a CONSERVATIVE sequence ####
coefs_third2_anti_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.neutral:recsys.ai - attitude.neutral:recsys.ac" &
           layer3_specificoutcome != "overall")


coefs_third2_anti_basecontrol$outcome = outcome_labels$outcome[match(coefs_third2_anti_basecontrol$layer3_specificoutcome,
                                                                     outcome_labels$specificoutcome)]

coefs_third2_anti_basecontrol$family = outcome_labels$family[match(coefs_third2_anti_basecontrol$layer3_specificoutcome,
                                                                   outcome_labels$specificoutcome)]

coefs_third2_anti_basecontrol <- mutate(coefs_third2_anti_basecontrol,
                                        family = factor(family,levels = c("Policy Attitudes\n(unit scale, + is more conservative)"
                                                                          ),ordered = T))


coefs_third2_anti_basecontrol <- coefs_third2_anti_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_third2_anti_basecontrol):1
  )

writeLines(as.character(round(100*abs(filter(coefs_third2_anti_basecontrol,layer3_specificoutcome=="pro_fraction_chosen")$est),0)),con = "../results/beta_recsys_pro_fraction_chosen_third2_antiseed.tex",sep="%")
writeLines(as.character(round(filter(coefs_third2_anti_basecontrol,layer1_hypothesisfamily=="gunpolicy")$est,2)),con = "../results/beta_recsys_mwindex_third2_antiseed.tex",sep="%")
writeLines(as.character(round(filter(coefs_third2_anti_basecontrol,layer1_hypothesisfamily=="gunpolicy")$est + qnorm(0.975)*filter(coefs_third2_anti_basecontrol,layer1_hypothesisfamily=="gunpolicy")$se,2)),con = "../results/cihi_recsys_mwindex_third2_antiseed.tex",sep="%")
writeLines(as.character(round(filter(coefs_third2_anti_basecontrol,layer1_hypothesisfamily=="gunpolicy")$est + qnorm(0.025)*filter(coefs_third2_anti_basecontrol,layer1_hypothesisfamily=="gunpolicy")$se,2)),con = "../results/cilo_recsys_mwindex_third2_antiseed.tex",sep="%")

#### THE effect of INCREASING vs. CONSTANT assignment among MODERATE participants assigned to a CONSERVATIVE sequence ####
(coefplot_third2_anti_basecontrol <- ggplot(filter(coefs_third2_anti_basecontrol),aes(y=plotorder)) +
    geom_errorbarh(aes(xmin=ci_lo_95,xmax=ci_hi_95),height=0,lwd=0.5) +
    geom_errorbarh(aes(xmin=ci_lo_90,xmax=ci_hi_90),height=0,lwd=1) +
    geom_point(aes(x=est),size=1.5) +
    geom_vline(xintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_y_continuous("",
                       breaks = coefs_third2_anti_basecontrol$plotorder,labels = coefs_third2_anti_basecontrol$outcome) +
    scale_x_continuous("Increasing Conservative vs. Constant Conservative seed \namong Moderates \n(95% and 90% CIs)") +
    coord_cartesian(xlim=c(-0.2,0.2)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"))
)
ggsave(coefplot_third2_anti_basecontrol,
       filename = "../results/coefplot_third2_anti_basecontrol.png",width=5,height=8)


#### THE effect of CONSERVATIVE vs. LIBERAL assignment among MODERATE participants assigned to an INCREASING sequence ####
coefs_third2_31_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.neutral:recsys.ai - attitude.neutral:recsys.pi" &
           layer3_specificoutcome != "overall")


coefs_third2_31_basecontrol$outcome = outcome_labels$outcome[match(coefs_third2_31_basecontrol$layer3_specificoutcome,
                                                                   outcome_labels$specificoutcome)]

coefs_third2_31_basecontrol$family = outcome_labels$family[match(coefs_third2_31_basecontrol$layer3_specificoutcome,
                                                                 outcome_labels$specificoutcome)]

coefs_third2_31_basecontrol <- mutate(coefs_third2_31_basecontrol,
                                      family = factor(family,levels = c("Policy Attitudes\n(unit scale, + is more conservative)"
                                                                        ),ordered = T))


coefs_third2_31_basecontrol <- coefs_third2_31_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_third2_31_basecontrol):1
  )
writeLines(as.character(round(100*abs(filter(coefs_third2_31_basecontrol,layer3_specificoutcome=="pro_fraction_chosen")$est),0)),con = "../results/beta_seed_pro_fraction_chosen_third2_31.tex",sep="%")


#### THE effect of CONSERVATIVE vs. LIBERAL assignment among MODERATE participants assigned to an INCREASING sequence ####
(coefplot_third2_31_basecontrol <- ggplot(filter(coefs_third2_31_basecontrol),aes(y=plotorder)) +
    geom_errorbarh(aes(xmin=ci_lo_95,xmax=ci_hi_95),height=0,lwd=0.5) +
    geom_errorbarh(aes(xmin=ci_lo_90,xmax=ci_hi_90),height=0,lwd=1) +
    geom_point(aes(x=est),size=1.5) +
    geom_vline(xintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_y_continuous("",
                       breaks = coefs_third2_31_basecontrol$plotorder,labels = coefs_third2_31_basecontrol$outcome) +
    scale_x_continuous("Conservative vs. Liberal seed assignment among Moderates\n with Increasing assignment\n(95% and 90% CIs)") +
    coord_cartesian(xlim=c(-0.2,0.2)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"))
)
ggsave(coefplot_third2_31_basecontrol,
       filename = "../results/coefplot_third2_31_basecontrol.png",width=5,height=8)

#### THE effect of CONSERVATIVE vs. LIBERAL assignment among MODERATE participants assigned to an CONSTANT sequence ####
coefs_third2_22_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.neutral:recsys.ac - attitude.neutral:recsys.pc" &
           layer3_specificoutcome != "overall")

coefs_third2_22_basecontrol$outcome = outcome_labels$outcome[match(coefs_third2_22_basecontrol$layer3_specificoutcome,
                                                                   outcome_labels$specificoutcome)]

coefs_third2_22_basecontrol$family = outcome_labels$family[match(coefs_third2_22_basecontrol$layer3_specificoutcome,
                                                                 outcome_labels$specificoutcome)]

coefs_third2_22_basecontrol <- mutate(coefs_third2_22_basecontrol,
                                      family = factor(family,levels = c(#"Platform Interaction",
                                                                        "Policy Attitudes\n(unit scale, + is more conservative)"
                                                                        #"Media Trust\n(unit scale, + is more trusting)",
                                                                        #"Affective Polarization\n(unit scale, + is greater polarization)"
                                                                        ),ordered = T))

#### THE effect of CONSERVATIVE vs. LIBERAL assignment among MODERATE participants assigned to an CONSTANT sequence ####
coefs_third2_22_basecontrol <- coefs_third2_22_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_third2_22_basecontrol):1
  )
writeLines(as.character(round(100*abs(filter(coefs_third2_22_basecontrol,layer3_specificoutcome=="pro_fraction_chosen")$est),0)),con = "../results/beta_seed_pro_fraction_chosen_third2_22.tex",sep="%")


(coefplot_third2_22_basecontrol <- ggplot(filter(coefs_third2_22_basecontrol),aes(y=plotorder)) +
    geom_errorbarh(aes(xmin=ci_lo_95,xmax=ci_hi_95),height=0,lwd=0.5) +
    geom_errorbarh(aes(xmin=ci_lo_90,xmax=ci_hi_90),height=0,lwd=1) +
    geom_point(aes(x=est),size=1.5) +
    geom_vline(xintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_y_continuous("",
                       breaks = coefs_third2_22_basecontrol$plotorder,labels = coefs_third2_22_basecontrol$outcome) +
    scale_x_continuous("Conservative vs. Liberal seed assignment among Moderates\n with Constant assignment\n(95% and 90% CIs)") +
    coord_cartesian(xlim=c(-0.2,0.2)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"))
)

rm(list = ls())
