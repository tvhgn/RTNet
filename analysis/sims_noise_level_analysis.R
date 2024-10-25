library(dplyr)
library(papaja)
library(ggridges)
library(ggplot2)
library(gridExtra)
library(viridis)
library(runjags)
library(bayesplot)

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

# Show histogram of confidence distribution across models
ggplot(data=df, aes(x=confidence, fill=model)) +
  geom_histogram(color='black',alpha=1, position='dodge') + 
  scale_fill_viridis_d(option="inferno", begin=color_begin, end=color_end) +
  xlab("Confidence") + 
  ylab("Frequency") +
  labs(fill="Model")

# Show histogram of confidence distribution across decision outcome
ggplot(data=df, aes(x=confidence, fill=correct)) +
  geom_histogram(color='black',alpha=1, position='dodge') + 
  scale_fill_viridis_d(option="inferno", begin=color_begin, end=color_end) +
  xlab("Confidence") + 
  ylab("Frequency") +
  labs(fill="Decision")

# Show ridgeplot across noise levels and separate by model
ridges_model <- df %>%
  ggplot(aes(x = confidence, y = factor(evidence), fill = model)) +
  geom_density_ridges(alpha = 0.7) +  # Add transparency for better visibility
  scale_fill_viridis_d(name="Model", option = "inferno",begin=color_begin, end = color_end) +
  xlab("Confidence") +
  ylab("Evidence level")
# Display the plots
print(ridges_model)

# Show ridgeplot across noise levels and separate by decision outcome
ridges_outcome <- df %>%
  ggplot(aes(x = confidence, y = factor(evidence), fill = correct)) +
    geom_density_ridges(alpha = 0.7) +  # Add fill aesthetic for better visibility
    scale_fill_viridis_d(name="Decision", labels=c("Error", "Correct"), option = "inferno",begin=color_begin, end = color_end) +
    xlab("Confidence") +
    ylab("Evidence level")
# Display the plots
print(ridges_outcome)

# Create function for getting the statistics
get_summary_stats <- function(data){
  # A summary per evidence level and grouped by decision outcome
  summarized_decision <- data %>%
    
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
  
# Summary statistics for correct and incorrect choices
data_summary <- get_summary_stats(data=df)

# Get a sample of datapoints per noise level to plot on the graph
set.seed(42)  
sampled_data <- df %>%
  #mutate(correct=true.label==choice) %>%
  group_by(model, evidence) %>%
  sample_n(size = 100) %>%  # Sample x points per noise level
  left_join(data_summary$by_decision %>% select(model, evidence, acc), by = c("model","evidence"))
  

# Create a combined plot for all models 
ggplot(data=data_summary$by_decision, aes(x=evidence, y=mean_conf, color = correct)) +
  geom_errorbar(aes(ymin=mean_conf-sem_conf, ymax=mean_conf+sem_conf, width=.025)) +
  geom_line() +
  geom_jitter(data = sampled_data, aes(x=evidence, y=confidence, color = correct), 
              width = 0.01, height = 0, size = 2, alpha = 0.1) +
  scale_color_viridis_d(name="Decision", labels=c("Error", "Correct"), option="inferno", begin=color_begin, end=color_end) +
  xlab("Evidence level") +
  ylab("Confidence") +
  facet_wrap(~model)  # Create separate plots for each model

# Get a sample to reduce computation time
plot_data_sample <- df %>%
  group_by(model) %>%
  sample_n(size=20000)

# Fit logistic regression model to data
# log_mod <- glm(formula= correct~evidence, data=plot_data_sample, family=binomial(link="logit"))
# summary(log_mod)
# 
# # Get log regress line plot
# b0 <- as.numeric(log_mod$coefficients[1])
# b1 <- as.numeric(log_mod$coefficients[2])
# evidence_lvls <- seq(from=0, to=1, by=0.01)
# glm_accs <- exp(b0+b1*evidence_lvls)/(1+exp(b0+b1*evidence_lvls))
# df_pred <- data.frame(evidence=evidence_lvls, acc=glm_accs) %>%
#   mutate(logits=log(acc/(1-acc)))

# Bayesian Logistic regression to model accuracy from evidence level.
# For both low and high confidence.

# Path to model for loading
# Pick model
for (mod in RTNet_models){
  jags_model_path <- file.path("..", "data", "JAGS", paste("model_",mod,"_jags_output_th_", th, ".RData", sep=""))
  # load model
  load(jags_model_path)
  
  # Extract parameter samples from posterior
  param_samples <- combine.mcmc(output)
  b0 <- as.numeric(param_samples[,1])
  b1 <- as.numeric(param_samples[,2])
  b2 <- as.numeric(param_samples[,3])
  b3 <- as.numeric(param_samples[,4])
  
  # Means
  b0_m <- mean(b0)
  b1_m <- mean(b1)
  b2_m <- mean(b2)
  b3_m <- mean(b3)
  
  # filter data by model
  acc_data <- data_summary$by_confidence %>%
    filter(model==mod)
  
  # Confidence group specific accuracies
  acc_low_conf <- acc_data[acc_data$confidence_group=="low",]
  acc_high_conf <- acc_data[acc_data$confidence_group=="high",]
  
  # Plot the Bayesian regression lines
  plot(x=acc_low_conf$evidence, xlab= "Evidence level",
       y=acc_low_conf$acc, ylab = "Accuracy", cex=1, pch=16,
       col='darkblue', ylim=c(0,1))
  # add title
  title(paste("Model", mod))
  
  # Add high confidence curve
  points(acc_high_conf$evidence, acc_high_conf$acc, col="red", pch=16)
  
  # Plot sample of regression lines
  smp <- sample(nrow(param_samples), 500)
  # Curve for low confidence
  invisible(lapply(smp, function(i) curve(exp(b0[i] + b1[i]*x)/(1+exp(b0[i] + b1[i]*x)), col=adjustcolor("purple",alpha=.1), add=TRUE )))
  # Curve for high confidence
  invisible(lapply(smp, function(i) curve(exp(b0[i] + b2[i] +b1[i]*x + b3[i]*x)/(1+exp(b0[i] + b2[i] + b1[i]*x + b3[i]*x)), col=adjustcolor("orange",alpha=.1), add=TRUE )))
  # Mean curve for low confidence
  curve(exp(b0_m+b1_m*x)/(1+exp(b0_m+b1_m*x)), col="darkblue", add=TRUE, lwd=2)
  # Mean curve for high confidence
  curve(exp(b0_m+b2_m+b1_m*x+b3_m*x)/(1+exp(b0_m+b2_m+b1_m*x+b3_m*x)), col="red", add=TRUE, lwd=2)
  
  # Add a legend to the plot
  legend("bottomright", 
         legend=c("High confidence (mean)", "Low confidence (mean)", "High confidence (uncertainty)", "Low confidence (uncertainty)","High confidence (data)", "Low confidence (data)"),
         col=c("red", "darkblue", adjustcolor("orange", alpha=.5), adjustcolor("purple", alpha=.5), 'red', "darkblue"), 
         lwd=c(2, 2, 1, 1, NA, NA), 
         pch=c(NA, NA, NA, NA, 16, 16),
         bty="n", 
         cex=0.8)
}


# acc-ev high low confidence
ggplot(data=data_summary$by_confidence, aes(x=evidence, y=acc, color=confidence_group)) +
  geom_line() +
  geom_point() +
  scale_color_viridis_d(name="Confidence", labels=c("High", "Low"), option="inferno", begin=color_begin, end=color_end) +
  xlab("Evidence level") +
  ylab("Accuracy") +
  facet_wrap(~model)

# evidence - RT Plot
ggplot(data=data_summary$by_decision, aes(x=evidence, y=mean_RT, color=correct)) +
  geom_line() + 
  geom_errorbar(aes(ymin=mean_RT-sem_RT, ymax=mean_RT+sem_RT, width=.025)) +
  geom_point() +
  scale_color_viridis_d(name="Decision", labels=c("Error", "Correct"), option="inferno", begin=color_begin, end=color_end) +
  xlab("Evidence level") + 
  ylab("RT") +
  facet_wrap(~model)

# Accuracy - confidence plot
ggplot(data=data_summary$by_evidence, aes(x=mean_conf, y=acc, color=model)) +
  geom_line() + 
  #geom_errorbar(aes(ymin=mean_conf-sem_conf, ymax=mean_conf+sem_conf, width=.025)) +
  geom_point() +
  scale_color_viridis_d(option="inferno", begin=color_begin, end=color_end) +
  xlab("Mean Confidence") + 
  ylab("Accuracy")


# confidence - RT plot
ggplot(data=plot_data_sample, aes(x=confidence, y=rt, color=model)) +
  geom_jitter(height=0.1, alpha=0.1) +
  geom_smooth(method=loess, se=TRUE) +
  scale_color_viridis_d(option="inferno", begin=color_begin, end=color_end) +
  xlab("Confidence") +
  ylab("RT") +
  labs(title=paste("Sample of all simulations (n =", nrow(plot_data_sample), ")"))



# TODO: RT distribution



