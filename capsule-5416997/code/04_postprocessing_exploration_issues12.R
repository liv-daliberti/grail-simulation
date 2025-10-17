cat(rep('=', 80),
    '\n\n',
    'OUTPUT FROM: 04_postprocessing_exploration_issues12.R',
    '\n\n',
    sep = ''
    )

## YouTube Algorithms and Minimum Wage Opinions
## Data collected May-June 2022 via MTurk/CloudResearch

## Preamble ----------------------------
library(tidyverse)
library(janitor)
library(lubridate)
library(stargazer)
library(broom)
library(patchwork)

# plotting w/ custom colors (optional)
red_mit = '#A31F34'
red_light = '#A9606C'
blue_mit = '#315485'
grey_light= '#C2C0BF'
grey_dark = '#8A8B8C'
black = '#353132'
vpurple = "#440154FF"
vyellow = "#FDE725FF"
vgreen = "#21908CFF"

## edited 13 june 2024 at request of reviewers ---------------------------------

understanding_1 <-
  read_csv('../results/intermediate data/gun control (issue 1)/guncontrol_understanding_basecontrol_pretty.csv') %>%
  mutate(
    layer2_treatmentcontrast = recode(
      layer2_treatmentcontrast,
      "31 pro - 22 pro" = "con 31 - con 22",
      "anti 31 - anti 22" = "lib 31 - lib 22",
      "31 neutral anti - 22 neutral anti" = "neutral lib 31 - neutral lib 22",
      "22 neutral pro - 22 neutral anti" = "neutral con 22 - neutral lib 22",
      "31 neutral pro - 31 neutral anti" = "neutral con 31 - neutral lib 31",
      "31 neutral pro - 22 neutral pro" = "neutral con 31 - neutral con 22"
    )
  )


understanding_2 <-
  read_csv('../results/intermediate data/minimum wage (issue 2)/understanding_basecontrol_pretty.csv')
understanding_2 <- understanding_2 %>%
  mutate(
    layer2_treatmentcontrast = recode(
      layer2_treatmentcontrast,
      "31 pro - 22 pro" = "con 31 - con 22",
      "anti 31 - anti 22" = "lib 31 - lib 22",
      "31 neutral anti - 22 neutral anti" = "neutral lib 31 - neutral lib 22",
      "22 neutral anti - 22 neutral pro" = "neutral con 22 - neutral lib 22",
      "31 neutral anti - 31 neutral pro" = "neutral con 31 - neutral lib 31",
      "31 neutral pro - 22 neutral pro" = "neutral con 31 - neutral con 22"
    )
  )


understanding_3 <- read_csv('../results/intermediate data/minimum wage (issue 2)/understanding_basecontrol_pretty_yg.csv')
understanding_3 <- understanding_3 %>%
  mutate(
    layer2_treatmentcontrast = recode(
      layer2_treatmentcontrast,
      "31 pro - 22 pro" = "con 31 - con 22",
      "anti 31 - anti 22" = "lib 31 - lib 22",
      "31 neutral anti - 22 neutral anti" = "neutral lib 31 - neutral lib 22",
      "22 neutral anti - 22 neutral pro" = "neutral con 22 - neutral lib 22",
      "31 neutral anti - 31 neutral pro" = "neutral con 31 - neutral lib 31",
      "31 neutral pro - 22 neutral pro" = "neutral con 31 - neutral con 22"
    )
  )

understanding_1$Study <- 1
understanding_2$Study <- 2
understanding_3$Study <- 3

understanding <- rbind(understanding_1,
                       understanding_2,
                       understanding_3
)
understanding$Study <- factor(understanding$Study,
                              levels = 3:1,
                              labels = c('Minimum Wage\n(YouGov)',
                                         'Minimum Wage\n(MTurk)',
                                         'Gun Control\n(MTurk)'
                              )
)

understanding <- understanding %>%
  mutate(outcome =
           recode(layer3_specificoutcome,
                  'right_to_own_importance_w2' = 'Question 1:\nRight to own more important than regulation (Gun Control)\nRestricts business freedom to set policy (Minimum Wage)',
                  'concealed_safe_w2' = 'Question 2:\nMore concealed carry makes US safer (Gun Control)\nRaising hurts low-income workers (Minimum Wage)',
                  'mw_restrict_w2' = 'Question 1:\nRight to own more important than regulation (Gun Control)\nRestricts business freedom to set policy (Minimum Wage)',
                  'mw_help_w2' = 'Question 2:\nMore concealed carry makes US safer (Gun Control)\nRaising hurts low-income workers (Minimum Wage)'
           )
  )

understanding <- understanding %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.999)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se
  )

understanding <- understanding %>%
  mutate(
    contrast = ifelse(
      layer2_treatmentcontrast %in% c("neutral con 31 - neutral lib 31",
                                      "neutral con 22 - neutral lib 22"
      ),
      yes = 'seed',
      no = 'algorithm'
    )
  )

understanding$layer2_treatmentcontrast <- factor(
  understanding$layer2_treatmentcontrast,
  levels = c('lib 31 - lib 22',
             'neutral lib 31 - neutral lib 22',
             'neutral con 31 - neutral con 22',
             'con 31 - con 22',
             'neutral con 31 - neutral lib 31',
             'neutral con 22 - neutral lib 22'
  ),
  labels = c('Liberal respondents,\nliberal seed',
             'Moderate respondents,\nliberal seed',
             'Moderate respondents,\nconservative seed',
             'Conservative respondents,\nconservative seed',
             'Moderate respondents,\n3/1 algorithm',
             'Moderate respondents,\n2/2 algorithm'
  ),
  ordered = TRUE
)

understanding_plot_algo <- ggplot(
  understanding %>% filter(contrast == 'algorithm'),
  aes(x = layer2_treatmentcontrast,
      group = Study,
      color = p.adj < 0.05
  )
) +
  geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95),
                position=position_dodge(width=0.5),
                width=0,
                lwd=0.5
  ) +
  geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90),
                position=position_dodge(width=0.5),
                width=0,
                lwd=1
  ) +
  geom_point(aes(y=est,shape=Study),
             position=position_dodge(width=0.5),
             size=2
  ) +
  geom_hline(yintercept = 0,lty=2) +
  facet_wrap( ~ outcome,scales="free") +
  scale_color_manual(breaks=c(F,T),values = c("black","blue"),guide="none") +
  coord_flip(ylim=c(-0.1,0.2)) +
  theme_bw(base_family = "sans") +
  theme(strip.background = element_rect(fill="white"),legend.position = "none") +
  ylab('Treatment effect of 3/1 vs. 2/2 algorithm (95% and 90% CIs)') +
  xlab(NULL)
understanding_plot_algo


understanding_plot_seed <- ggplot(
  understanding %>% filter(contrast == 'seed'),
  aes(x = layer2_treatmentcontrast,
      group = Study,
      color = p.adj < 0.05
  )
) +
  geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95),
                position=position_dodge(width=0.5),
                width=0,
                lwd=0.5
  ) +
  geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90),
                position=position_dodge(width=0.5),
                width=0,
                lwd=1
  ) +
  geom_point(aes(y=est,shape=Study),
             position=position_dodge(width=0.5),
             size=2
  ) +
  geom_hline(yintercept = 0,lty=2) +
  facet_wrap(~ outcome,scales="free") +
  scale_color_manual(breaks=c(F,T),values = c("black","blue"),guide="none") +
  coord_flip(ylim=c(-0.1,0.2)) +
  theme_bw(base_family = "sans") +
  theme(strip.background = element_rect(fill="white"),legend.position = "bottom",legend.margin = margin(0,0,0,-3,"lines")) +
  ylab('Treatment effect of conservative seed vs. liberal seed video (95% and 90% CIs)') +
  xlab(NULL)

understanding_plot <- (understanding_plot_algo / understanding_plot_seed) +
  plot_layout(heights = c(2, 1))

ggsave(understanding_plot,
       filename = "../results/understanding_3studies.png",width=12,height=8.5)

## Base-control Figures ----------------------------------------------------

coefs_basecontrol_guns <- read_csv("../results/intermediate data/gun control (issue 1)/guncontrol_padj_basecontrol_pretty.csv") %>%
  mutate(est = case_when(layer3_specificoutcome=="pro_fraction_chosen" ~ -1*est,
                         layer3_specificoutcome!="pro_fraction_chosen" ~ est),
         layer2_treatmentcontrast = dplyr::recode(layer2_treatmentcontrast,
                                                  "pro 31 - pro 22"="con 31 - con 22",
                                                  "anti 31 - anti 22"="lib 31 - lib 22",
                                                  "neutral anti 31 - neutral anti 22"="neutral lib 31 - neutral lib 22",
                                                  "neutral pro 22 - neutral anti 22"="neutral con 22 - neutral lib 22",
                                                  "neutral pro 31 - neutral anti 31"="neutral con 31 - neutral lib 31",
                                                  "neutral pro 31 - neutral pro 22"="neutral con 31 - neutral con 22"
         ))
coefs_basecontrol <- read_csv("../results/intermediate data/minimum wage (issue 2)/padj_basecontrol_pretty.csv") %>%
  mutate(layer2_treatmentcontrast = dplyr::recode(layer2_treatmentcontrast,
                                                  "pro 31 - pro 22"="lib 31 - lib 22",
                                                  "anti 31 - anti 22"="con 31 - con 22",
                                                  "neutral anti 31 - neutral anti 22"="neutral con 31 - neutral con 22",
                                                  "neutral anti 22 - neutral pro 22"="neutral con 22 - neutral lib 22",
                                                  "neutral anti 31 - neutral pro 31"="neutral con 31 - neutral lib 31",
                                                  "neutral pro 31 - neutral pro 22"="neutral lib 31 - neutral lib 22"
  ))
coefs_basecontrol_yg <- read_csv("../results/intermediate data/minimum wage (issue 2)/padj_basecontrol_pretty_yg.csv") %>%
  mutate(layer2_treatmentcontrast = dplyr::recode(layer2_treatmentcontrast,
                                                  "pro 31 - pro 22"="lib 31 - lib 22",
                                                  "anti 31 - anti 22"="con 31 - con 22",
                                                  "neutral anti 31 - neutral anti 22"="neutral con 31 - neutral con 22",
                                                  "neutral anti 22 - neutral pro 22"="neutral con 22 - neutral lib 22",
                                                  "neutral anti 31 - neutral pro 31"="neutral con 31 - neutral lib 31",
                                                  "neutral pro 31 - neutral pro 22"="neutral lib 31 - neutral lib 22"
  ))
coefs_basecontrol <- bind_rows(mutate(coefs_basecontrol_guns,Sample="Gun Control\n(MTurk)"),
                               mutate(coefs_basecontrol,Sample="Minimum Wage\n(MTurk)"),
                               mutate(coefs_basecontrol_yg,Sample="Minimum Wage\n(YouGov)")) %>%
  mutate(Sample = factor(Sample,levels=c("Minimum Wage\n(YouGov)","Minimum Wage\n(MTurk)","Gun Control\n(MTurk)"),ordered=T)) %>%
  mutate(layer1_hypothesisfamily = recode(layer1_hypothesisfamily,
                                          "gunpolicy"="policy",
                                          "mwpolicy"="policy"),
         layer3_specificoutcome = recode(layer3_specificoutcome,
                                         "gun_index_w2"="policyindex",
                                         "mw_index_w2"="policyindex"))

# look at significant effects:
coefs_basecontrol %>% filter(!str_detect(layer2_treatmentcontrast,"neutral") & p.adj < .05 & layer3_specificoutcome != 'overall')


coefs_basecontrol %>% filter(str_detect(layer2_treatmentcontrast,"neutral") & p.adj < .05 & layer3_specificoutcome != 'overall' &
                               ((str_detect(layer2_treatmentcontrast,"lib") & !str_detect(layer2_treatmentcontrast,"con")) |
                                  !(str_detect(layer2_treatmentcontrast,"lib") & str_detect(layer2_treatmentcontrast,"con"))))

outcome_labels <- data.frame(outcome = c(
  "Liberal videos\nchosen (fraction)",
  "Likes & saves\nminus dislikes (#)",
  "Total watch\ntime (hrs)",
  "Policy\nindex",
  "Trust in\nmajor news",
  "Trust in\nYouTube",
  "Never fabrication\nby major news",
  "Never fabrication\nby YouTube",
  "Perceived intelligence",
  "Feeling thermometer",
  "Comfort as friend"),
  specificoutcome = c(
    "pro_fraction_chosen",
    "positive_interactions",
    "platform_duration",
    "policyindex",
    "trust_majornews_w2",
    "trust_youtube_w2",
    "fabricate_majornews_w2",
    "fabricate_youtube_w2",
    "affpol_smart_w2",
    "affpol_ft_w2",
    "affpol_comfort_w2"),
  family = c(
    rep("Platform Interaction",3),
    rep("Policy Attitudes\n(unit scale, + is more conservative)",1),
    rep("Media Trust\n(unit scale, + is more trusting)",4),
    rep("Affective Polarization\n(unit scale, + is greater polarization)",3))
)

##### Liberals #####
coefs_third1_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "lib 31 - lib 22" &
           layer3_specificoutcome != "overall")

coefs_third1_basecontrol$outcome = outcome_labels$outcome[match(coefs_third1_basecontrol$layer3_specificoutcome,
                                                                outcome_labels$specificoutcome)]


coefs_third1_basecontrol$family = outcome_labels$family[match(coefs_third1_basecontrol$layer3_specificoutcome,
                                                              outcome_labels$specificoutcome)]


coefs_third1_basecontrol <- mutate(coefs_third1_basecontrol,
                                   family = factor(family,
                                                   levels = c(
                                                     "Policy Attitudes\n(unit scale, + is more conservative)",
                                                     "Platform Interaction",
                                                     "Media Trust\n(unit scale, + is more trusting)",
                                                     "Affective Polarization\n(unit scale, + is greater polarization)"),ordered = T))

## manipulate to get all unit scales:
coefs_third1_basecontrol$est[coefs_third1_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third1_basecontrol$est[coefs_third1_basecontrol$layer3_specificoutcome=="platform_duration"]/3600
coefs_third1_basecontrol$se[coefs_third1_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third1_basecontrol$se[coefs_third1_basecontrol$layer3_specificoutcome=="platform_duration"]/3600

coefs_third1_basecontrol$est[coefs_third1_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third1_basecontrol$est[coefs_third1_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100
coefs_third1_basecontrol$se[coefs_third1_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third1_basecontrol$se[coefs_third1_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100

coefs_third1_basecontrol <- coefs_third1_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.999)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = rep((nrow(coefs_third1_basecontrol)/3):1,3),
         alpha = ifelse(p.adj<0.05, T, F),
         alpha = as.logical(alpha),
         alpha = replace_na(alpha,F),
         Sample_color = as.character(Sample),
         Sample_color = replace(Sample_color,alpha==F,"insig")
  )
tabyl(coefs_third1_basecontrol,Sample_color)

(coefplot_third1_basecontrol <- ggplot(filter(coefs_third1_basecontrol),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5,alpha=0.25) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1,alpha=0.25) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3,alpha=0.25) +
    geom_text(data=filter(coefs_third1_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third1_basecontrol$plotorder,labels = coefs_third1_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all liberal seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),
          legend.position = "none",
    )
)
ggsave(coefplot_third1_basecontrol,
       filename = "../results/coefplot_third1_basecontrol_3studies.png",width=5,height=8.5)
ggsave(coefplot_third1_basecontrol,
       filename = "../results/coefplot_third1_basecontrol_3studies.pdf",width=5,height=8.5)

(coefplot_third1_basecontrol_empty <- ggplot(filter(coefs_third1_basecontrol),aes(x=plotorder,group=Sample,alpha=alpha,col=Sample)) +
    geom_blank(aes(ymin=ci_lo_95,ymax=ci_hi_95),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_blank(aes(ymin=ci_lo_90,ymax=ci_hi_90),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_blank(aes(y=est,shape=Sample),position=position_dodge(width=0.5),size=3) +
    geom_blank(data=filter(coefs_third1_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third1_basecontrol$plotorder,labels = coefs_third1_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all liberal seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position = "none")
)
ggsave(coefplot_third1_basecontrol_empty,
       filename = "../results/coefplot_third1_basecontrol_empty_3studies.png",width=5,height=8.5)

(coefplot_third1_basecontrol_3studies_toptwo <- ggplot(filter(coefs_third1_basecontrol,layer1_hypothesisfamily %in% c("policy","platform")),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5,alpha=0.25) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1,alpha=0.25) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3,alpha=0.25) +
    geom_text(data=filter(coefs_third1_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third1_basecontrol$plotorder,labels = coefs_third1_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all liberal seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),
          legend.position = "none",
    )
)
ggsave(coefplot_third1_basecontrol_3studies_toptwo,
       filename = "../results/coefplot_third1_basecontrol_3studies_toptwo.png",width=5,height=4.75)
ggsave(coefplot_third1_basecontrol_3studies_toptwo,
       filename = "../results/coefplot_third1_basecontrol_3studies_toptwo.pdf",width=5,height=4.75)


##### Conservatives #####

coefs_third3_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "con 31 - con 22" &
           layer3_specificoutcome != "overall")

coefs_third3_basecontrol$outcome = outcome_labels$outcome[match(coefs_third3_basecontrol$layer3_specificoutcome,
                                                                outcome_labels$specificoutcome)]

coefs_third3_basecontrol$family = outcome_labels$family[match(coefs_third3_basecontrol$layer3_specificoutcome,
                                                              outcome_labels$specificoutcome)]

coefs_third3_basecontrol <- mutate(coefs_third3_basecontrol,
                                   family = factor(family,levels = c("Policy Attitudes\n(unit scale, + is more conservative)","Platform Interaction","Media Trust\n(unit scale, + is more trusting)","Affective Polarization\n(unit scale, + is greater polarization)"),ordered = T))

## manipulate to get all unit scales:
coefs_third3_basecontrol$est[coefs_third3_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third3_basecontrol$est[coefs_third3_basecontrol$layer3_specificoutcome=="platform_duration"]/3600
coefs_third3_basecontrol$se[coefs_third3_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third3_basecontrol$se[coefs_third3_basecontrol$layer3_specificoutcome=="platform_duration"]/3600

coefs_third3_basecontrol$est[coefs_third3_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third3_basecontrol$est[coefs_third3_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100
coefs_third3_basecontrol$se[coefs_third3_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third3_basecontrol$se[coefs_third3_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100

coefs_third3_basecontrol <- coefs_third3_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.999)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = rep((nrow(coefs_third3_basecontrol)/3):1,3),
         alpha = ifelse(p.adj<0.05, T, F),
         alpha = as.logical(alpha),
         alpha = replace_na(alpha,F),
         Sample_color = as.character(Sample),
         Sample_color = replace(Sample_color,alpha==F,"insig")
  )


(coefplot_third3_basecontrol <- ggplot(filter(coefs_third3_basecontrol),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3) +
    geom_text(data=filter(coefs_third3_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third3_basecontrol$plotorder,labels = coefs_third3_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all conservative seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)

ggsave(coefplot_third3_basecontrol,
       filename = "../results/coefplot_third3_basecontrol_3studies.png",width=5,height=8.5)
ggsave(coefplot_third3_basecontrol,
       filename = "../results/coefplot_third3_basecontrol_3studies.pdf",width=5,height=8.5)

(coefplot_third3_basecontrol_empty <- ggplot(filter(coefs_third3_basecontrol),aes(x=plotorder,group=Sample,col=ifelse(p.adj<0.05,T,F))) +
    geom_blank(aes(ymin=ci_lo_95,ymax=ci_hi_95),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_blank(aes(ymin=ci_lo_90,ymax=ci_hi_90),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_blank(aes(y=est,shape=Sample),position=position_dodge(width=0.5),size=2) +
    geom_blank(data=filter(coefs_third3_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third3_basecontrol$plotorder,labels = coefs_third3_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all conservative seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    coord_flip(ylim=c(-0.17,0.17)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)
ggsave(coefplot_third3_basecontrol_empty,
       filename = "../results/coefplot_third3_basecontrol_empty_3studies.png",width=5,height=8.5)

(coefplot_third3_basecontrol_toptwo <- ggplot(filter(coefs_third3_basecontrol,layer1_hypothesisfamily %in% c("policy","platform")),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3) +
    geom_text(data=filter(coefs_third3_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third3_basecontrol$plotorder,labels = coefs_third3_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all conservative seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)

ggsave(coefplot_third3_basecontrol_toptwo,
       filename = "../results/coefplot_third3_basecontrol_3studies_toptwo.png",width=5,height=4.75)
ggsave(coefplot_third3_basecontrol_toptwo,
       filename = "../results/coefplot_third3_basecontrol_3studies_toptwo.pdf",width=5,height=4.75)


##### Moderates (algorithm) #####

coefs_third2_pro_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "neutral lib 31 - neutral lib 22" &
           layer3_specificoutcome != "overall")

coefs_third2_pro_basecontrol$outcome = outcome_labels$outcome[match(coefs_third2_pro_basecontrol$layer3_specificoutcome,
                                                                    outcome_labels$specificoutcome)]

coefs_third2_pro_basecontrol$family = outcome_labels$family[match(coefs_third2_pro_basecontrol$layer3_specificoutcome,
                                                                  outcome_labels$specificoutcome)]

coefs_third2_pro_basecontrol <- mutate(coefs_third2_pro_basecontrol,
                                       family = factor(family,levels = c("Policy Attitudes\n(unit scale, + is more conservative)","Platform Interaction","Media Trust\n(unit scale, + is more trusting)","Affective Polarization\n(unit scale, + is greater polarization)"),ordered = T))

## manipulate to get all unit scales:
coefs_third2_pro_basecontrol$est[coefs_third2_pro_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third2_pro_basecontrol$est[coefs_third2_pro_basecontrol$layer3_specificoutcome=="platform_duration"]/3600
coefs_third2_pro_basecontrol$se[coefs_third2_pro_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third2_pro_basecontrol$se[coefs_third2_pro_basecontrol$layer3_specificoutcome=="platform_duration"]/3600

coefs_third2_pro_basecontrol$est[coefs_third2_pro_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third2_pro_basecontrol$est[coefs_third2_pro_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100
coefs_third2_pro_basecontrol$se[coefs_third2_pro_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third2_pro_basecontrol$se[coefs_third2_pro_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100

coefs_third2_pro_basecontrol <- coefs_third2_pro_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.999)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = rep((nrow(coefs_third2_pro_basecontrol)/3):1,3),
         alpha = ifelse(p.adj<0.05, T, F),
         alpha = as.logical(alpha),
         alpha = replace_na(alpha,F),
         Sample_color = as.character(Sample),
         Sample_color = replace(Sample_color,alpha==F,"insig")
  )
writeLines(as.character(abs(round(filter(coefs_third2_pro_basecontrol,layer3_specificoutcome=="platform_duration" & Sample=="Minimum Wage\n(YouGov)")$est*60,1))),
           con = "../results/beta_minutes_recsys_duration_third2_proseed_study3.tex",sep="%")

(coefplot_third2_pro_basecontrol <- ggplot(filter(coefs_third2_pro_basecontrol),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3) +
    geom_text(data=filter(coefs_third2_pro_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third2_pro_basecontrol$plotorder,labels = coefs_third2_pro_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all liberal seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)
ggsave(coefplot_third2_pro_basecontrol,
       filename = "../results/coefplot_third2_pro_basecontrol_3studies.png",width=5,height=8.5)
ggsave(coefplot_third2_pro_basecontrol,
       filename = "../results/coefplot_third2_pro_basecontrol_3studies.pdf",width=5,height=8.5)

(coefplot_third2_pro_basecontrol_empty <- ggplot(filter(coefs_third2_pro_basecontrol),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_blank(aes(ymin=ci_lo_95,ymax=ci_hi_95),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_blank(aes(ymin=ci_lo_90,ymax=ci_hi_90),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_blank(aes(y=est,shape=Sample),position=position_dodge(width=0.5),size=3) +
    geom_blank(data=filter(coefs_third2_pro_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third2_pro_basecontrol$plotorder,labels = coefs_third2_pro_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all liberal seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)
ggsave(coefplot_third2_pro_basecontrol_empty,
       filename = "../results/coefplot_third2_pro_basecontrol_empty_3studies.png",width=5,height=8.5)

(coefplot_third2_pro_basecontrol_toptwo <- ggplot(filter(coefs_third2_pro_basecontrol,layer1_hypothesisfamily %in% c("policy","platform")),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3) +
    geom_text(data=filter(coefs_third2_pro_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third2_pro_basecontrol$plotorder,labels = coefs_third2_pro_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all liberal seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)
ggsave(coefplot_third2_pro_basecontrol_toptwo,
       filename = "../results/coefplot_third2_pro_basecontrol_3studies_toptwo.png",width=5,height=4.75)
ggsave(coefplot_third2_pro_basecontrol_toptwo,
       filename = "../results/coefplot_third2_pro_basecontrol_3studies_toptwo.pdf",width=5,height=4.75)

coefs_third2_anti_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "neutral con 31 - neutral con 22" &
           layer3_specificoutcome != "overall")

coefs_third2_anti_basecontrol$outcome = outcome_labels$outcome[match(coefs_third2_anti_basecontrol$layer3_specificoutcome,
                                                                     outcome_labels$specificoutcome)]

coefs_third2_anti_basecontrol$family = outcome_labels$family[match(coefs_third2_anti_basecontrol$layer3_specificoutcome,
                                                                   outcome_labels$specificoutcome)]

coefs_third2_anti_basecontrol <- mutate(coefs_third2_anti_basecontrol,
                                        family = factor(family,levels = c("Policy Attitudes\n(unit scale, + is more conservative)","Platform Interaction","Media Trust\n(unit scale, + is more trusting)","Affective Polarization\n(unit scale, + is greater polarization)"),ordered = T))

## manipulate to get all unit scales:
coefs_third2_anti_basecontrol$est[coefs_third2_anti_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third2_anti_basecontrol$est[coefs_third2_anti_basecontrol$layer3_specificoutcome=="platform_duration"]/3600
coefs_third2_anti_basecontrol$se[coefs_third2_anti_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third2_anti_basecontrol$se[coefs_third2_anti_basecontrol$layer3_specificoutcome=="platform_duration"]/3600

coefs_third2_anti_basecontrol$est[coefs_third2_anti_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third2_anti_basecontrol$est[coefs_third2_anti_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100
coefs_third2_anti_basecontrol$se[coefs_third2_anti_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third2_anti_basecontrol$se[coefs_third2_anti_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100

coefs_third2_anti_basecontrol <- coefs_third2_anti_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.999)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = rep((nrow(coefs_third2_anti_basecontrol)/3):1,3),
         alpha = ifelse(p.adj<0.05, T, F),
         alpha = as.logical(alpha),
         alpha = replace_na(alpha,F),
         Sample_color = as.character(Sample),
         Sample_color = replace(Sample_color,alpha==F,"insig")
  )

writeLines(as.character(abs(round(filter(coefs_third2_anti_basecontrol,layer3_specificoutcome=="platform_duration" & Sample=="Gun Control\n(MTurk)")$est*60,1))),
           con = "../results/beta_minutes_recsys_duration_third2_antiseed_study1.tex",sep="%")

(coefplot_third2_anti_basecontrol <- ggplot(filter(coefs_third2_anti_basecontrol),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3) +
    geom_text(data=filter(coefs_third2_anti_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third2_anti_basecontrol$plotorder,labels = coefs_third2_anti_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all conservative seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)
ggsave(coefplot_third2_anti_basecontrol,
       filename = "../results/coefplot_third2_anti_basecontrol_3studies.png",width=5,height=8.5)
ggsave(coefplot_third2_anti_basecontrol,
       filename = "../results/coefplot_third2_anti_basecontrol_3studies.pdf",width=5,height=8.5)

(coefplot_third2_anti_basecontrol_empty <- ggplot(filter(coefs_third2_anti_basecontrol),aes(x=plotorder,group=Sample,col=ifelse(p.adj<0.05,T,F))) +
    geom_blank(aes(ymin=ci_lo_95,ymax=ci_hi_95),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_blank(aes(ymin=ci_lo_90,ymax=ci_hi_90),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_blank(aes(y=est,shape=Sample),position=position_dodge(width=0.5),size=2) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third2_anti_basecontrol$plotorder,labels = coefs_third2_anti_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all conservative seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)
ggsave(coefplot_third2_anti_basecontrol_empty,
       filename = "../results/coefplot_third2_anti_basecontrol_empty_3studies.png",width=5,height=8.5)

(coefplot_third2_anti_basecontrol_toptwo <- ggplot(filter(coefs_third2_anti_basecontrol,layer1_hypothesisfamily %in% c("policy","platform")),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3) +
    geom_text(data=filter(coefs_third2_anti_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third2_anti_basecontrol$plotorder,labels = coefs_third2_anti_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of 3/1 vs. 2/2\nalgorithm, all conservative seed\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)
ggsave(coefplot_third2_anti_basecontrol_toptwo,
       filename = "../results/coefplot_third2_anti_basecontrol_3studies_toptwo.png",width=5,height=4.75)
ggsave(coefplot_third2_anti_basecontrol_toptwo,
       filename = "../results/coefplot_third2_anti_basecontrol_3studies_toptwo.pdf",width=5,height=4.75)


##### Moderates (seed) #####
coefs_third2_31_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "neutral con 31 - neutral lib 31" &
           layer3_specificoutcome != "overall")

coefs_third2_31_basecontrol$outcome = outcome_labels$outcome[match(coefs_third2_31_basecontrol$layer3_specificoutcome,
                                                                   outcome_labels$specificoutcome)]

coefs_third2_31_basecontrol$family = outcome_labels$family[match(coefs_third2_31_basecontrol$layer3_specificoutcome,
                                                                 outcome_labels$specificoutcome)]

coefs_third2_31_basecontrol <- mutate(coefs_third2_31_basecontrol,
                                      family = factor(family,levels = c("Policy Attitudes\n(unit scale, + is more conservative)","Platform Interaction","Media Trust\n(unit scale, + is more trusting)","Affective Polarization\n(unit scale, + is greater polarization)"),ordered = T))

## manipulate to get all unit scales:
coefs_third2_31_basecontrol$est[coefs_third2_31_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third2_31_basecontrol$est[coefs_third2_31_basecontrol$layer3_specificoutcome=="platform_duration"]/3600
coefs_third2_31_basecontrol$se[coefs_third2_31_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third2_31_basecontrol$se[coefs_third2_31_basecontrol$layer3_specificoutcome=="platform_duration"]/3600

coefs_third2_31_basecontrol$est[coefs_third2_31_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third2_31_basecontrol$est[coefs_third2_31_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100
coefs_third2_31_basecontrol$se[coefs_third2_31_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third2_31_basecontrol$se[coefs_third2_31_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100

coefs_third2_31_basecontrol <- coefs_third2_31_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.999)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = rep((nrow(coefs_third2_31_basecontrol)/3):1,3),
         alpha = ifelse(p.adj<0.05, T, F),
         alpha = as.logical(alpha),
         alpha = replace_na(alpha,F),
         Sample_color = as.character(Sample),
         Sample_color = replace(Sample_color,alpha==F,"insig")
  )

dummy_df <- data.frame(family=c("Platform Interaction","Platform Interaction"),est=c(-0.5,0.5),plotorder=c(9,9),Sample=c("Gun Control\n(MTurk)","Gun Control\n(MTurk)"),alpha=c(FALSE,FALSE)) %>% mutate(family=factor(family))

(coefplot_third2_31_basecontrol <- ggplot(filter(coefs_third2_31_basecontrol),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3) +
    geom_blank(data=dummy_df,aes(y=est)) +
    geom_text(data=filter(coefs_third2_31_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    # facet_grid(rows = "family",scales="free",space = "free_y",switch = "y") +
    scale_x_continuous("",
                       breaks = coefs_third2_31_basecontrol$plotorder,labels = coefs_third2_31_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of conservative seed vs.\nliberal seed video, all 3/1 algorithm\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    # coord_flip(ylim=c(-0.3,0.3)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)
ggsave(coefplot_third2_31_basecontrol,
       filename = "../results/coefplot_third2_31_basecontrol_3studies.png",width=5,height=8.5)
ggsave(coefplot_third2_31_basecontrol,
       filename = "../results/coefplot_third2_31_basecontrol_3studies.pdf",width=5,height=8.5)

(coefplot_third2_31_basecontrol_empty <- ggplot(filter(coefs_third2_31_basecontrol),aes(x=plotorder,group=Sample,col=ifelse(p.adj<0.05,T,F))) +
    geom_blank(aes(ymin=ci_lo_95,ymax=ci_hi_95),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_blank(aes(ymin=ci_lo_90,ymax=ci_hi_90),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_blank(aes(y=est,shape=Sample),position=position_dodge(width=0.5),size=2) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third2_31_basecontrol$plotorder,labels = coefs_third2_31_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of conservative seed vs.\nliberal seed video, all 3/1 algorithm\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip(ylim=c(-0.3,0.3)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="bottom",legend.margin = margin(0,0,0,-3,"lines"))
)
ggsave(coefplot_third2_31_basecontrol_empty,
       filename = "../results/coefplot_third2_31_basecontrol_empty_3studies.png",width=5,height=8.5)


# create DF to set axis limits:
dummy_df <- data.frame(family=c("Platform Interaction","Platform Interaction"),est=c(-0.5,0.5),plotorder=c(9,9),Sample=c("Gun Control\n(MTurk)","Gun Control\n(MTurk)"),alpha=c(FALSE,FALSE)) %>% mutate(family=factor(family))

(coefplot_third2_31_basecontrol_toptwo <- ggplot(filter(coefs_third2_31_basecontrol,layer1_hypothesisfamily %in% c("policy","platform")),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3) +
    geom_blank(data=dummy_df,aes(y=est)) +
    geom_text(data=filter(coefs_third2_31_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    # facet_grid(rows = "family",scales="free",space = "free_y",switch = "y") +
    scale_x_continuous("",
                       breaks = coefs_third2_31_basecontrol$plotorder,labels = coefs_third2_31_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of conservative seed vs.\nliberal seed video, all 3/1 algorithm\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    # coord_flip(ylim=c(-0.4,0.4)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none",plot.margin = margin(5,10,5,5))
)
ggsave(coefplot_third2_31_basecontrol_toptwo,
       filename = "../results/coefplot_third2_31_basecontrol_3studies_toptwo.png",width=5,height=4.75)
ggsave(coefplot_third2_31_basecontrol_toptwo,
       filename = "../results/coefplot_third2_31_basecontrol_3studies_toptwo.pdf",width=5,height=4.75)



coefs_third2_22_basecontrol <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "neutral con 22 - neutral lib 22" &
           layer3_specificoutcome != "overall")

coefs_third2_22_basecontrol$outcome = outcome_labels$outcome[match(coefs_third2_22_basecontrol$layer3_specificoutcome,
                                                                   outcome_labels$specificoutcome)]

coefs_third2_22_basecontrol$family = outcome_labels$family[match(coefs_third2_22_basecontrol$layer3_specificoutcome,
                                                                 outcome_labels$specificoutcome)]

coefs_third2_22_basecontrol <- mutate(coefs_third2_22_basecontrol,
                                      family = factor(family,levels = c("Policy Attitudes\n(unit scale, + is more conservative)","Platform Interaction","Media Trust\n(unit scale, + is more trusting)","Affective Polarization\n(unit scale, + is greater polarization)"),ordered = T))

## manipulate to get all unit scales:
coefs_third2_22_basecontrol$est[coefs_third2_22_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third2_22_basecontrol$est[coefs_third2_22_basecontrol$layer3_specificoutcome=="platform_duration"]/3600
coefs_third2_22_basecontrol$se[coefs_third2_22_basecontrol$layer3_specificoutcome=="platform_duration"] <- coefs_third2_22_basecontrol$se[coefs_third2_22_basecontrol$layer3_specificoutcome=="platform_duration"]/3600

coefs_third2_22_basecontrol$est[coefs_third2_22_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third2_22_basecontrol$est[coefs_third2_22_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100
coefs_third2_22_basecontrol$se[coefs_third2_22_basecontrol$layer3_specificoutcome=="affpol_ft_w2"] <- coefs_third2_22_basecontrol$se[coefs_third2_22_basecontrol$layer3_specificoutcome=="affpol_ft_w2"]/100

coefs_third2_22_basecontrol <- coefs_third2_22_basecontrol %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.999)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = rep((nrow(coefs_third2_22_basecontrol)/3):1,3),
         alpha = ifelse(p.adj<0.05, T, F),
         alpha = as.logical(alpha),
         alpha = replace_na(alpha,F),
         Sample_color = as.character(Sample),
         Sample_color = replace(Sample_color,alpha==F,"insig")
  )

(coefplot_third2_22_basecontrol <- ggplot(filter(coefs_third2_22_basecontrol),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3) +
    geom_text(data=filter(coefs_third2_22_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third2_22_basecontrol$plotorder,labels = coefs_third2_22_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of conservative seed vs.\nliberal seed video, all 2/2 algorithm\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)
ggsave(coefplot_third2_22_basecontrol,
       filename = "../results/coefplot_third2_22_basecontrol_3studies.png",width=5,height=8.5)
ggsave(coefplot_third2_22_basecontrol,
       filename = "../results/coefplot_third2_22_basecontrol_3studies.pdf",width=5,height=8.5)

(coefplot_third2_22_basecontrol_empty <- ggplot(filter(coefs_third2_22_basecontrol),aes(x=plotorder,group=Sample,col=ifelse(p.adj<0.05,T,F))) +
    geom_blank(aes(ymin=ci_lo_95,ymax=ci_hi_95),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_blank(aes(ymin=ci_lo_90,ymax=ci_hi_90),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_blank(aes(y=est,shape=Sample),position=position_dodge(width=0.5),size=2) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third2_22_basecontrol$plotorder,labels = coefs_third2_22_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of conservative seed vs.\nliberal seed video, all 2/2 algorithm\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip(ylim=c(-0.6,0.6)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="bottom",legend.margin = margin(0,0,0,-3,"lines"))
)
ggsave(coefplot_third2_22_basecontrol_empty,
       filename = "../results/coefplot_third2_22_basecontrol_empty_3studies.png",width=5,height=8.5)

(coefplot_third2_22_basecontrol_toptwo <- ggplot(filter(coefs_third2_22_basecontrol,layer1_hypothesisfamily %in% c("policy","platform")),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3) +
    geom_text(data=filter(coefs_third2_22_basecontrol,layer1_hypothesisfamily=="policy"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~family,ncol=1,scales="free") +
    scale_x_continuous("",
                       breaks = coefs_third2_22_basecontrol$plotorder,labels = coefs_third2_22_basecontrol$outcome) +
    scale_y_continuous("Treatment effect of conservative seed vs.\nliberal seed video, all 2/2 algorithm\n(95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip() +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none")
)
ggsave(coefplot_third2_22_basecontrol_toptwo,
       filename = "../results/coefplot_third2_22_basecontrol_3studies_toptwo.png",width=5,height=4.75)
ggsave(coefplot_third2_22_basecontrol_toptwo,
       filename = "../results/coefplot_third2_22_basecontrol_3studies_toptwo.pdf",width=5,height=4.75)

##### All respondents, attitudinal DV only #####
coefs_policyindex <- filter(coefs_third2_22_basecontrol,layer1_hypothesisfamily=="policy") %>% mutate(contrast="Seed, 2/2",subset="Moderates") %>%
  bind_rows(filter(coefs_third2_31_basecontrol,layer1_hypothesisfamily=="policy") %>% mutate(contrast="Seed, 3/1",subset="Moderates")) %>%
  bind_rows(filter(coefs_third2_pro_basecontrol,layer1_hypothesisfamily=="policy") %>% mutate(contrast="Algorithm, lib. seed",subset="Moderates (liberal seed)")) %>%
  bind_rows(filter(coefs_third2_anti_basecontrol,layer1_hypothesisfamily=="policy") %>% mutate(contrast="Algorithm, cons. seed",subset="Moderates  (conservative seed)")) %>%
  bind_rows(filter(coefs_third1_basecontrol,layer1_hypothesisfamily=="policy") %>% mutate(contrast="Algorithm, lib. seed",subset="Liberals  (liberal seed)")) %>%
  bind_rows(filter(coefs_third3_basecontrol,layer1_hypothesisfamily=="policy") %>% mutate(contrast="Algorithm, cons. seed",subset="Conservatives  (conservative seed)")) %>%
  mutate(subset = factor(subset,levels=c("Liberals  (liberal seed)","Conservatives  (conservative seed)","Moderates (liberal seed)","Moderates  (conservative seed)"),ordered = T))

(coefplot_policyindex_basecontrol <- ggplot(filter(coefs_policyindex,str_detect(contrast,"Algorithm")),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=3) +
    geom_text(data=filter(coefs_policyindex,subset=="Liberals  (liberal seed)"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~subset,ncol=2,scales="free") +
    scale_x_continuous("",breaks = 8,labels="") +
    scale_y_continuous("Treatment effect of more extreme 3/1 vs. 2/2\nalgorithm on policy index (95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip(ylim=c(-0.11,0.11)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="bottom",legend.margin = margin(0,0,0,-3,"lines"),
          axis.ticks.y = element_blank())
)
ggsave(coefplot_policyindex_basecontrol,
       filename = "../results/coefplot_policyindex_basecontrol_3studies.png",width=4.5,height=4.5)

(coefplot_policyindex_seed_basecontrol <- ggplot(filter(coefs_policyindex,str_detect(contrast,"Seed")),aes(x=plotorder,group=Sample,col=Sample,alpha=alpha)) +
    geom_errorbar(aes(ymin=ci_lo_95,ymax=ci_hi_95,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=0.5) +
    geom_errorbar(aes(ymin=ci_lo_90,ymax=ci_hi_90,col=Sample_color),position=position_dodge(width=0.5),width=0,lwd=1) +
    geom_point(aes(y=est,shape=Sample,col=Sample_color),position=position_dodge(width=0.5),size=2) +
    geom_text(data=filter(coefs_policyindex,contrast=="Seed, 2/2"),aes(y=est+0.006,label=Sample),alpha=1,position=position_dodge(width=0.5),size=3) +
    geom_hline(yintercept = 0,lty=2) +
    facet_wrap(~contrast,ncol=2,scales="free") +
    scale_x_continuous("",breaks = 8,labels="") +
    scale_y_continuous("Treatment effect of conservative vs. liberal\nseed on policy index (95% and 90% CIs)") +
    scale_color_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)","insig"),values=c(vgreen,red_mit,blue_mit,"black")) +
    scale_shape_manual("Study:",breaks = c("Gun Control\n(MTurk)","Minimum Wage\n(MTurk)","Minimum Wage\n(YouGov)"),values=c(16,17,18)) +
    scale_alpha_manual(breaks=c(F,T),values=c(0.25,1)) +
    coord_flip(ylim=c(-0.11,0.11)) +
    theme_bw(base_family = "sans") +
    theme(strip.background = element_rect(fill="white"),legend.position="none",
          axis.ticks.y = element_blank())
)
ggsave(coefplot_policyindex_seed_basecontrol,
       filename = "../results/coefplot_policyindex_seed_basecontrol_3studies.png",width=4.5,height=2.5)

rm(list = ls())
