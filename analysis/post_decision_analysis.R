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
df <- read.csv(file.path("..", "results", paste("model_",mod,"_post_decision_sims_3_3.csv", sep="")))

# Plot CoM: correct-to-incorrect, incorrect-to-correct
# First do pre-processing
df <- df %>%
  mutate(model=mod) %>%
  # Determine percentile groups
  mutate(conf_perc_group=ntile(confidence, 10)) %>% # Create groups based on deciles.
  
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
  geom_bar(fill='lightblue') +
  xlab("Change of Mind") +
  ylab("Count") +
  scale_fill_viridis_d(option="inferno", begin=color_begin, end=color_end)


# Contour plot
ev_lvls <-  seq(0, 1, 0.1)
# Folders to store pngs and gif file
gif_path <- file.path("..", "results", "plots", "CoM", "gifs")
png_path <- file.path(gif_path, "pngs")

for (ev in ev_lvls){
  print(paste("Saving contourplot for level:", ev))
  # Filter the data for evidence level
  filtered_df <- df %>%
    filter(evidence==round(ev, 1) & CoM!="No Change")
  
  cont_plot <- ggplot(data=filtered_df, aes(x=confidence, y=confidence_post)) +
    geom_density_2d_filled(alpha=1) + 
    geom_abline(intercept = 0, slope = 1, linetype = "dotted", linewidth=1, color = "white") +
    scale_fill_viridis_d(option="inferno") +
    xlim(c(0, 1)) +
    ylim(c(0, 1)) +
    xlab("Initial Confidence") +
    ylab("Post-decision confidence") +
    facet_grid(cols=vars(CoM)) +
    labs(title = paste("Evidence level =", ev))
  file_n <- paste("contour_plot_ev_", sub(".", "_", ev, fixed=TRUE), ".png", sep="")
  ggsave(filename = file_n, plot=cont_plot, path=png_path, width=5000, height=2500, units="px")
}

# Convert to gif
png_files <- list.files(png_path, pattern="^contour.*.png$", full.names=TRUE)
gifski(png_files, gif_file=file.path(gif_path,"contour_animation.gif"), width=1200, height=600, delay=1)

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

# anim <- animate(cont_plot, width=1000, height=500, renderer = gifski_renderer(), duration=10)
# anim_save("confidence_changes_across_evidence_lvls.gif", animation=anim, path=gif_path)

# Create plot of confidence differences among groups
comparisons <- list(c("Corrected", "No Change"), c("Spoilt", "No Change"), c("Corrected", "Spoilt"))

for (ev in ev_lvls){
  print(paste("Saving violin plot for level:", ev))
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
  
  ggsave(file.path(png_path, paste("violin_ev_", sub(".", "_", ev, fixed=TRUE), ".png", sep="")), violins, width=6, height=5)
}

# Convert to gif
png_files <- list.files(png_path, pattern="^violin.*.png$", full.names=TRUE)
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
  filter(evidence==0) %>%
  group_by(choice) %>%
  summarize(
    n=n()
  )

print(df_ev_lvl)

df_com_llh <- df %>%
  group_by(conf_perc_group) %>%
  summarise(
    n=n(),
    n_com = sum(CoM!="No Change"),
    n_spoilt = sum(CoM=="Spoilt"),
    llh_com = n_com/n,
    llh_spoilt = n_spoilt/n
  )

#View(df_com_llh)

ggplot(data=df_com_llh, aes(x=conf_perc_group, y=llh_com))+
  geom_point(color='green', shape=16) +
  geom_point(aes(y=llh_spoilt), color='purple', shape=17) +
  xlab("Confidence decile") +
  ylab("Proportion of trials")

# MNIST id's with high initial confidence
df_high_conf <- df %>%
  filter(conf_perc_group>7) %>%
  sample_n(size=10)

df_low_conf <- df %>%
  filter(conf_perc_group<3) %>%
  sample_n(size=10)

df_high_conf$mnist_index
df_low_conf$mnist_index


