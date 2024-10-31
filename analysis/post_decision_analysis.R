library(dplyr)
library(tidyr)
library(papaja)
library(ggridges)
library(ggplot2)
library(viridis)
library(car)
library(ggpubr)
library(gganimate)
library(gifski)

theme_set(theme_apa())

# Color parameters
color_begin <- 0.2
color_end <- 0.85

# Read data
RTNet_models <- c("01", "36")
mod <- RTNet_models[1]
df <- read.csv(file.path("..", "results", paste("model_",mod,"_post_decision_sims_3.csv", sep="")))

# Plot CoM: correct-to-incorrect, incorrect-to-correct
# First do pre-processing
df <- df %>%
  mutate(model=mod) %>%
  # Create column with binary factor 'correct'
  mutate(correct=true.label==choice) %>%
  mutate(correct_post=(true.label==choice_post)) %>%

  # Create column to indicate if CoM occurred and what type if so.
  mutate(CoM=case_when(correct==correct_post ~ "No Change",
                       correct<correct_post ~ "Corrected",
                       correct>correct_post ~ "Spoilt")) %>%
  
  # Create a column with the confidence difference (conf_post - conf)
  mutate(c_diff=confidence_post-confidence) %>%
  
  # Factorize
  mutate(correct=factor(correct)) %>%
  mutate(correct_post=factor(correct_post)) %>%
  mutate(choice=factor(choice)) %>%
  mutate(true.label=factor(true.label)) %>%
  mutate(CoM=factor(CoM)) %>%

  # Create column with post-choice correct
  mutate(correct_post=factor(true.label==choice_post))

n_evidence <- nrow(df[df$evidence==0,])

# Create a summary dataframe that contains the proportion of CoM levels across evidence levels
evidence_summary <- df %>%
  group_by(evidence, CoM) %>%
  summarize(
    prop=n() / n_evidence
  )

# Show how proportions change across evidence levels
ggplot(data=evidence_summary, aes(x=evidence, y=prop, color=CoM)) +
  geom_line() +
  geom_point() +
  scale_fill_viridis_d(option="inferno", begin=color_begin, end=color_end) +
  xlab("Evidence Level") +
  ylab("Proportion")


# Show how proportions change across evidence levels, excluding no change group
ggplot(data=evidence_summary[evidence_summary$CoM!="No Change",], aes(x=evidence, y=prop, color=CoM)) +
  geom_line() +
  geom_point() +
  scale_fill_viridis_d(option="inferno", begin=color_begin, end=color_end) +
  xlab("Evidence Level") +
  ylab("Proportion")

# Create histogram 
ggplot(data=df, aes(x=CoM)) +
  geom_bar() +
  scale_fill_viridis_d(option="inferno", begin=color_begin, end=color_end)

# Contour plot
# Confidence vs. post-decision confidence improved vs. worsened confidence groups
# cont_plot <- ggplot(data=df[df$CoM!="No Change",], aes(x=confidence, y=confidence_post)) +
#   geom_density_2d_filled(alpha=1) +  # Adjust alpha for visibility
#   geom_abline(intercept = 0, slope = 1, linetype = "dotted", linewidth=1, color = "white") +
#   scale_fill_viridis_d(name="Density", option="inferno") +
#   xlab("Confidence") +
#   ylab("Post-decision confidence") +
#   facet_grid(cols=vars(CoM)) +
#   transition_states(evidence, transition_length=1, state_length=10)+
#   labs(title = "Evidence level: {closest_state}")
# 
# anim_path <- file.path("..", "data", "plots", "gifs")
# anim <- animate(cont_plot, width=1000, height=500, renderer = gifski_renderer(), duration=10)
# anim_save("confidence_changes_across_evidence_lvls.gif", animation=anim, path=anim_path)

# Create plot of confidence differences among groups
comparisons <- list(c("Corrected", "No Change"), c("Spoilt", "No Change"), c("Corrected", "Spoilt"))
ev_lvls <-  seq(0, 1, 0.1)
for (ev in ev_lvls){
  print(ev)
  filtered_df <- df %>%
    filter(evidence==round(ev, 1))
  
  violins <- ggplot(data=filtered_df, aes(x=CoM, y=c_diff, fill=CoM)) +
    geom_violin()+
    stat_summary(fun=mean, geom="point", color="black") +
    scale_fill_viridis_d(name="Change of Mind",option="inferno", begin=color_begin+0.1, end=color_end) +
    #facet_grid(~evidence) +
    ylim(c(-1,1)) +
    xlab("Group") +
    ylab("Post confidence - Confidence") +
    labs(title=paste("Evidence level =", ev))
  
  ggsave(file.path(anim_path, paste("violin_ev_", sub(".", "_", ev, fixed=TRUE), ".png", sep="")), violins, width=6, height=5)
}
# Convert to gif
gif_path <- file.path("..", "data", "plots", "gifs")
png_files <- list.files(file.path(gif_path, "pngs"), pattern=".png$", full.names=TRUE)
gifski(png_files, gif_file=file.path(gif_path,"violin_animation.gif"), width=800, height=600, delay=1)

# Test for variance homogeneity - Unequal variances among groups!
levene_test <- leveneTest(c_diff ~ CoM, data = df)
print(levene_test)

# Get summary statistics
summary_stats <- df %>%
  group_by(CoM) %>%
  summarize(
    n=n(),
    mean=mean(c_diff),
    std=sd(c_diff)
  )

print(summary_stats)


# Check 0 evidence predictions
df_ev_lvl <- df %>%
  filter(evidence==0.6) %>%
  group_by(choice) %>%
  summarize(
    n=n()
  )
  




  

