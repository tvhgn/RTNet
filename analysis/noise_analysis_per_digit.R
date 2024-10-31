library(dplyr)
library(papaja)
library(ggridges)
library(ggplot2)
library(gridExtra)
library(viridis)
library(car)
library(ggpubr)
library(gifski)

# Set standard theme
theme_set(theme_apa())

# Pick model
RTNet_models <- c("01", "36")

# Pick threshold to filter
th <- 3
# Color parameters
color_begin <- 0.2
color_end <- 0.85

# Load datafiles
df1 <- read.csv(file.path("..", "results", paste("model_",RTNet_models[1],"_evidence_level_sims.csv", sep=""))) %>%
  mutate(model=RTNet_models[1])

df2 <- read.csv(file.path("..", "results", paste("model_",RTNet_models[2],"_evidence_level_sims.csv", sep=""))) %>%
  mutate(model=RTNet_models[2])

df <- rbind(df1, df2) %>%
  # Filter threshold
  filter(threshold==th) %>%
  # Factorize
  mutate(model=factor(model)) %>%
  mutate(choice=factor(choice)) %>%
  mutate(true.label=factor(true.label)) %>%
  # Create column with binary factor 'correct'
  mutate(correct=factor(true.label==choice))

# Evidence - accuracy plot across digits

# Create function for getting the statistics
get_summary_stats <- function(data, digit_filter){
  # A summary per evidence level and grouped by decision outcome
  summarized_decision <- data %>%
    # Filter by digit
    filter(true.label==digit_filter) %>%
    # Group by model, evidence level, as well as decision outcome
    group_by(model, evidence, correct) %>%
    summarize(
      mean_conf = mean(confidence),
      std_conf = sd(confidence),
      n = n(),
      sem_conf = std_conf/sqrt(n),
      mean_RT = mean(rt),
      std_RT = sd(rt),
      sem_RT = std_RT/sqrt(n)
    ) %>%
    
    group_by(evidence) %>%
    mutate(
      total_n = sum(n),
      acc = (sum(n[correct == TRUE]) / total_n)
    ) %>%
    ungroup()
  
  # A summary for just evidence level
  summarized_evidence <- data %>%
    # Determine accuracy and confidence per evidence level
    group_by(model, evidence) %>%
    summarize(
      n = n(), # total per evidence level
      n_correct = sum(correct==TRUE),
      acc = n_correct/n,
      mean_conf = mean(confidence),
      std_conf= sd(confidence),
      sem_conf = std_conf/sqrt(n)
    )
  
  # A summary which groups by high or low confidence
  summarized_confidence <- data %>%
    group_by(model) %>%
    mutate(
      confidence_group = factor(ifelse(confidence <= median(confidence), "low", "high"))
    ) %>%
    group_by(model, evidence, confidence_group) %>%
    summarize(
      n=n(),
      n_correct=sum(correct==TRUE),
      acc = n_correct/n,
      mean_conf=mean(confidence),
      std_conf=sd(confidence),
      sem_conf=std_conf/sqrt(n)
    )
  
  return (list(by_decision=summarized_decision,
               by_evidence=summarized_evidence,
               by_confidence=summarized_confidence))
}

for (digit in 0:9){
  # Summary statistics for correct and incorrect choices
  data_summary <- get_summary_stats(data=df, digit_filter=digit)
  
  # Get a sample of datapoints per noise level to plot on the graph
  set.seed(42)  
  sampled_data <- df %>%
    #mutate(correct=true.label==choice) %>%
    group_by(model, evidence) %>%
    sample_n(size = 100) %>%  # Sample x points per noise level
    left_join(data_summary$by_decision %>% select(model, evidence, acc), by = c("model","evidence"))
  
  
  # Create a combined plot for all models 
  ev_conf_plot <- ggplot(data=data_summary$by_decision, aes(x=evidence, y=mean_conf, color = correct)) +
    geom_errorbar(aes(ymin=mean_conf-sem_conf, ymax=mean_conf+sem_conf, width=.025)) +
    geom_line() +
    geom_jitter(data = sampled_data, aes(x=evidence, y=confidence, color = correct), 
                width = 0.01, height = 0, size = 2, alpha = 0.1) +
    scale_color_viridis_d(name="Decision", labels=c("Error", "Correct"), option="inferno", begin=color_begin, end=color_end) +
    xlab("Evidence level") +
    ylab("Confidence") +
    labs(title=paste("Digit =", digit)) +
    facet_wrap(~model)  # Create separate plots for each model
  # Save plot
  plot_file_dir <- file.path("..", "results", "plots", "digit_analysis")
  file_n <- paste("evidence_level_digit_", digit, ".png", sep="")
  ggsave(filename=file_n, plot=ev_conf_plot, path=plot_file_dir, width=12, height=5)
}

# Summary statistics based on digits
digits_summary <- df %>%
  group_by(true.label, correct) %>%
  summarize(
    n=n(),
    mean_conf=mean(confidence),
    std_conf=sd(confidence),
    sem_conf = std_conf/sqrt(n),
    mean_RT = mean(rt),
    std_RT = sd(rt),
    sem_RT = std_RT/sqrt(n)
  )
