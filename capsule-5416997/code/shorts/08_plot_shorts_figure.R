cat(rep('=', 80),
    '\n\n',
    'OUTPUT FROM: shorts/08_plot_shorts_figure.R',
    '\n\n',
    sep = ''
    )

library(tidyverse)
library(janitor)
library(lubridate)
library(stargazer)
library(broom)
library(psych)
library(ggtext)
library(ggplot2)

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


## MODEL RESULTS
coefs_basecontrol <- read_csv("../results/padj_basecontrol_pretty_ytrecs_may2024.csv")

outcome_labels <- data.frame(outcome = c("Minimum wage<br>index"),
                             specificoutcome = c("mw_index"),
                             family = c(rep("Policy Attitudes<br>(unit scale, + is more conservative)",1)))


# HYP 1
#### THE effect of INCREASING vs. CONSTANT assignment among LIBERAL participants ####
coefs_hyp1 <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.pro:recsys.pi - attitude.pro:recsys.pc" &
           layer3_specificoutcome != "overall")


coefs_hyp1$outcome = outcome_labels$outcome[match(coefs_hyp1$layer3_specificoutcome,
                                                                outcome_labels$specificoutcome)]

coefs_hyp1$family = outcome_labels$family[match(coefs_hyp1$layer3_specificoutcome,outcome_labels$specificoutcome)]

coefs_hyp1 <- mutate(coefs_hyp1,
                     family = factor(family,
                                     levels = c("Policy Attitudes<br>(unit scale, + is more conservative)"
                                                ),ordered = T))

coefs_hyp1 <- coefs_hyp1 %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_hyp1):1
  )

## HYP 2
#### THE effect of INCREASING vs. CONSTANT assignment among CONSERVATIVE participants ####
coefs_hyp2 <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.anti:recsys.ai - attitude.anti:recsys.ac" &
           layer3_specificoutcome != "overall")

coefs_hyp2$outcome = outcome_labels$outcome[match(coefs_hyp2$layer3_specificoutcome,
                                                                outcome_labels$specificoutcome)]

coefs_hyp2$family = outcome_labels$family[match(coefs_hyp2$layer3_specificoutcome,
                                                              outcome_labels$specificoutcome)]

coefs_hyp2 <- mutate(coefs_hyp2,
                                   family = factor(family,levels = c("Policy Attitudes<br>(unit scale, + is more conservative)"
                                   ),ordered = T))


coefs_hyp2 <- coefs_hyp2 %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_hyp2):1
  )

# HYP 3
#### THE effect of INCREASING vs. CONSTANT assignment among MODERATE participants assigned to a LIBERAL sequence ####
coefs_hyp3 <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.neutral:recsys.pi - attitude.neutral:recsys.pc" &
           layer3_specificoutcome != "overall")


coefs_hyp3$outcome = outcome_labels$outcome[match(coefs_hyp3$layer3_specificoutcome,
                                                                    outcome_labels$specificoutcome)]

coefs_hyp3$family = outcome_labels$family[match(coefs_hyp3$layer3_specificoutcome,
                                                                  outcome_labels$specificoutcome)]

coefs_hyp3 <- mutate(coefs_hyp3,
                                       family = factor(family,levels = c("Policy Attitudes<br>(unit scale, + is more conservative)"
                                       ),ordered = T))

coefs_hyp3 <- coefs_hyp3 %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_hyp3):1
  )

# HYP 4
#### THE effect of INCREASING vs. CONSTANT assignment among MODERATE participants assigned to a CONSERVATIVE sequence ####
coefs_hyp4 <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.neutral:recsys.ai - attitude.neutral:recsys.ac" &
           layer3_specificoutcome != "overall")


coefs_hyp4$outcome = outcome_labels$outcome[match(coefs_hyp4$layer3_specificoutcome,
                                                                     outcome_labels$specificoutcome)]

coefs_hyp4$family = outcome_labels$family[match(coefs_hyp4$layer3_specificoutcome,
                                                                   outcome_labels$specificoutcome)]

coefs_hyp4 <- mutate(coefs_hyp4,
                                        family = factor(family,levels = c("Policy Attitudes<br>(unit scale, + is more conservative)"
                                        ),ordered = T))


coefs_hyp4 <- coefs_hyp4 %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_hyp4):1
  )

# HYP 5
#### THE effect of CONSERVATIVE vs. LIBERAL assignment among MODERATE participants assigned to an INCREASING sequence ####
coefs_hyp5 <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.neutral:recsys.ai - attitude.neutral:recsys.pi" &
           layer3_specificoutcome != "overall")


coefs_hyp5$outcome = outcome_labels$outcome[match(coefs_hyp5$layer3_specificoutcome,
                                                                   outcome_labels$specificoutcome)]

coefs_hyp5$family = outcome_labels$family[match(coefs_hyp5$layer3_specificoutcome,
                                                                 outcome_labels$specificoutcome)]

coefs_hyp5 <- mutate(coefs_hyp5,
                     family = factor(family,levels = c("Policy Attitudes<br>(unit scale, + is more conservative)"
                                      ),ordered = T))


coefs_hyp5 <- coefs_hyp5 %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_hyp5):1
  )

# HYP 6
#### THE effect of CONSERVATIVE vs. LIBERAL assignment among MODERATE participants assigned to an CONSTANT sequence ####
coefs_hyp6 <- coefs_basecontrol %>%
  filter(layer2_treatmentcontrast == "attitude.neutral:recsys.ac - attitude.neutral:recsys.pc" &
           layer3_specificoutcome != "overall")

coefs_hyp6$outcome = outcome_labels$outcome[match(coefs_hyp6$layer3_specificoutcome,
                                                                   outcome_labels$specificoutcome)]

coefs_hyp6$family = outcome_labels$family[match(coefs_hyp6$layer3_specificoutcome,
                                                                 outcome_labels$specificoutcome)]

coefs_hyp6 <- mutate(coefs_hyp6,
                     family = factor(family,levels = c("Policy Attitudes<br>(unit scale, + is more conservative)"
                                       ),ordered = T))

#### THE effect of CONSERVATIVE vs. LIBERAL assignment among MODERATE participants assigned to an CONSTANT sequence ####
coefs_hyp6 <- coefs_hyp6 %>%
  mutate(ci_lo_99 = est + qnorm(0.001)*se,
         ci_hi_99 = est + qnorm(0.995)*se,
         ci_lo_95 = est + qnorm(0.025)*se,
         ci_hi_95 = est + qnorm(0.975)*se,
         ci_lo_90 = est + qnorm(0.05)*se,
         ci_hi_90 = est + qnorm(0.95)*se,
         plotorder = nrow(coefs_hyp6):1,
  )

# Combine all data frames into one
all_coefs <- bind_rows(
  mutate(coefs_hyp1, hypothesis = "**Increasing vs. Constant**<br>Liberal Seed<br>Liberal Ideologues", Sample="**Increasing vs. Constant**<br>Liberal Seed"),
  mutate(coefs_hyp2, hypothesis = "**Increasing vs. Constant**<br>Conservative Seed<br>Conservative Ideologues", Sample="**Increasing vs. Constant**<br>Conservative Seed"),
  mutate(coefs_hyp3, hypothesis = "**Increasing vs. Constant**<br>Liberal Seed<br>Moderates", Sample="**Increasing vs. Constant**<br>Liberal Seed"),
  mutate(coefs_hyp4, hypothesis = "**Increasing vs. Constant**<br>Conservative Seed<br>Moderates", Sample="**Increasing vs. Constant**<br>Conservative Seed"),
  mutate(coefs_hyp5, hypothesis = "**Conservative vs. Liberal**<br>Increasing Extremity<br>Moderates", Sample="**Conservative vs. Liberal**<br>Increasing Extremity"),
  mutate(coefs_hyp6, hypothesis = "**Conservative vs. Liberal**<br>Constant Extremity<br>Moderates", Sample="**Conservative vs. Liberal**<br>Constant Extremity")
)

# Define the order of hypotheses
hypothesis_order <- c("**Increasing vs. Constant**<br>Liberal Seed<br>Liberal Ideologues", 
                      "**Increasing vs. Constant**<br>Conservative Seed<br>Conservative Ideologues", 
                      "**Increasing vs. Constant**<br>Liberal Seed<br>Moderates", 
                      "**Increasing vs. Constant**<br>Conservative Seed<br>Moderates", 
                      "**Conservative vs. Liberal**<br>Increasing Extremity<br>Moderates", 
                      "**Conservative vs. Liberal**<br>Constant Extremity<br>Moderates")

# Reorder the factor levels
all_coefs$hypothesis <- factor(all_coefs$hypothesis, levels = hypothesis_order)

all_coefs <- all_coefs %>%
  mutate(
    attitude = case_when(
      row_number() == 1 ~ "Liberal Ideologues",
      row_number() == 2 ~ "Conservative Ideologues",
      TRUE ~ "Moderates"
    ),
    alpha = ifelse(p.adj<0.05, T, F),
    alpha = as.logical(alpha),
    alpha = replace_na(alpha,F),
    Sample_color = as.character(Sample),
    Sample_color = replace(Sample_color,alpha==F,"insig")
  )

all_coefs <- all_coefs %>%
  mutate(
    sign_color = case_when(
      ci_lo_95 < 0 & ci_hi_95 > 0 ~ grey_dark,  # black color code
      TRUE ~ "darkgreen"  # blue color code (or replace with your desired color code)
    )
  )

all_coefs <- all_coefs %>%
  mutate(
    attitude_color = case_when(
      attitude == "Liberal Ideologues" ~ blue_mit,
      attitude == "Conservative Ideologues" ~ red_mit,
      attitude == "Moderates" ~ "darkgreen"
    )
  )


all_coefs <- all_coefs %>%
  mutate(Sample = factor(Sample,levels=c("**Increasing vs. Constant**<br>Liberal Seed",
                                         "**Increasing vs. Constant**<br>Conservative Seed",
                                         "**Conservative vs. Liberal**<br>Increasing Extremity",
                                         "**Conservative vs. Liberal**<br>Constant Extremity"),
                         ordered=T)) #%>%
  #mutate(layer1_hypothesisfamily = recode(layer1_hypothesisfamily,
  #                                        "mwpolicy"="policy"),
  #       layer3_specificoutcome = recode(layer3_specificoutcome,
  #                                       "mw_index"="policyindex"))


# Create a data frame for attitude shapes
attitude_shapes <- data.frame(attitude = c("Liberal Ideologues", "Conservative Ideologues", "Moderates"))

# Plot the attitude shapes
attitude_bar <- ggplot(attitude_shapes, aes(x = attitude)) +
  geom_point(aes(shape = attitude), size = 3) +
  scale_shape_manual(values = c("Liberal Ideologues" = 16, "Conservative Ideologues" = 17, "Moderates" = 15)) +
  theme_void() +
  theme(legend.position = "none")

# Create a data frame for attitude shapes
attitude_shapes <- data.frame(attitude = c("Liberal Ideologues", "Conservative Ideologues", "Moderates"))

# Plot the attitude shapes
attitude_bar <- ggplot(attitude_shapes, aes(x = attitude)) +
  geom_point(aes(shape = attitude), size = 5) +
  scale_shape_manual(values = c("Liberal Ideologues" = 16, "Conservative Ideologues" = 17, "Moderates" = 15)) +
  theme_void() +
  theme(legend.position = "none")

# Plot
combined_plot <- ggplot(all_coefs, aes(x = est, y = Sample, group = attitude, shape = attitude)) +
  # 95% CI: Adjust alpha based on significance
  geom_errorbarh(aes(xmin = ci_lo_95, xmax = ci_hi_95, color = sign_color, alpha = 0.8), 
                 height = 0, lwd = 1, position = position_dodge(width = 0.8)) +
  
  # 90% CI: Adjust alpha based on significance
  geom_errorbarh(aes(xmin = ci_lo_90, xmax = ci_hi_90, color = sign_color, alpha = 0.8), 
                 height = 0, lwd = 1.5, position = position_dodge(width = 0.8)) +
  
  # Points: Adjust alpha directly for better visibility of insignificant shapes
  geom_point(aes(color = sign_color), 
             size = 4, position = position_dodge(width = 0.8), 
             alpha = ifelse(all_coefs$alpha, 1, 0.7)) +  # Make insignificant points more visible with 0.7 alpha
  
  # Labels: Adjust alpha based on significance
  geom_text(data = all_coefs, 
            aes(x = est, label = attitude, color = attitude_color), 
            alpha = 1, size = 6, 
            position = position_dodge(width = 0.8), vjust = -0.6) + 
  
  geom_vline(xintercept = 0, lty = 2) +
  facet_wrap(~ family, ncol = 1, scales = "free") +
  coord_cartesian(xlim = c(-0.06, 0.18), clip="off") +
  scale_x_continuous(" Minimum Wage Policy Effect Size\n(95% and 90% CIs)") +
  scale_color_identity() +  # Ensure that the color column is used directly
  labs(y = NULL) +  # Remove y-axis title
  theme_bw(base_family = "sans") +
  theme(strip.background = element_rect(fill = "white"),
        legend.position = "none",
        axis.text.y = element_markdown(color = "black", size=16),
        axis.title.x = element_markdown(color = "black", size=16),
        strip.text = element_markdown(size = 18)
  )
combined_plot
ggsave(combined_plot, filename = "../results/shorts_combined_intervals.pdf", width = 8.5, height = 5)
rm(list = ls())




