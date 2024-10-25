library(dplyr)
library(papaja)
library(ggridges)
library(ggplot2)
library(gridExtra)
library(viridis)
library(runjags)
library(bayesplot)

#### Data preparation ####
# Pick model
RTNet_models <- c("01", "36")
mod <- RTNet_models[1]

# Pick threshold to filter
th <- 3

# Load datafiles
df1 <- read.csv(file.path("results", paste("model_",RTNet_models[1],"_evidence_level_sims.csv", sep=""))) %>%
  mutate(model=RTNet_models[1]) %>%
  filter(threshold==th) %>%
  # Create column with binary confidence level
  mutate(confidence_group=ifelse(confidence <= median(confidence), 0, 1))

df2 <- read.csv(file.path("results", paste("model_",RTNet_models[2],"_evidence_level_sims.csv", sep=""))) %>%
  mutate(model=RTNet_models[2]) %>%
  filter(threshold==th) %>%
  # Create column with binary confidence level
  mutate(confidence_group=ifelse(confidence <= median(confidence), 0, 1))

df <- rbind(df1, df2) %>%
  # Factorize
  mutate(model=factor(model)) %>%
  mutate(choice=factor(choice)) %>%
  mutate(true.label=factor(true.label)) %>%
  # Create column with binary factor 'correct'
  mutate(correct=true.label==choice)
  
# Get a sample to reduce computation time
data_sample <- df %>%
  group_by(model) %>%
  sample_n(size=20000)

#### Bayesian logistic regression using JAGS ####
modelString <- "model{
  # Likelihood
  for (i in 1:n){
    y[i] ~ dbern(theta[i])
    logit(theta[i]) <- b0+b1*x1[i]+b2*x2[i] + b3*(x1[i]*x2[i])
  }
  
  # prior
  b0 ~ dnorm(0, 0.0001)
  b1 ~ dnorm(0, 0.0001)
  b2 ~ dnorm(0, 0.0001)
  b3 ~ dnorm(0, 0.0001)
  
}"
model_path <- file.path("data", "JAGS", "log_reg_model_conf_levels.txt")
writeLines(modelString, model_path)

# Prepare data
data_sample <- data_sample %>%
  filter(model==mod) # Filter by model, adjust as necessary
# Prepare datalist for JAGS
dataList <- list(y=as.numeric(data_sample$correct),
                 x1=data_sample$evidence,
                 x2=data_sample$confidence_group,
                 n=nrow(data_sample))

# Prepare model
# Model parameters
pars <- c("b0", "b1", "b2", "b3")
nChains <- 3
inits <- list(
  list("b0" = rnorm(1, 0, 10), "b1" = rnorm(1, 0, 10), "b2" = rnorm(1, 0, 10), "b3" = rnorm(1,0,10)),  # Chain 1
  list("b0" = rnorm(1, 0, 10), "b1" = rnorm(1, 0, 10), "b2" = rnorm(1, 0, 10), "b3" = rnorm(1,0,10)),  # Chain 2
  list("b0" = rnorm(1, 0, 10), "b1" = rnorm(1, 0, 10), "b2" = rnorm(1, 0, 10), "b3" = rnorm(1,0,10))   # Chain 3
)
samples <- 8000
burnin <- 1000
nAdapt <- 1000

# Path to model (use either for saving or loading)
jags_model_path <- file.path("data", "JAGS", paste("model_",mod,"_jags_output_th_", th, ".RData", sep=""))

# Run model or load previous model

set.seed(42) # Set seed
output <- run.jags(model=model_path,
                   monitor=pars,
                   data=dataList,
                   n.chains = nChains,
                   inits=inits,
                   burnin=burnin,
                   sample=samples,
                   adapt=nAdapt,
                   method="rjparallel"
)

# Save the output (to prevent long sampling time of about 10 minutes)
save(output, file=jags_model_path)

# MCMC diagnostics
mcmc_trace(output$mcmc, pars) # traceplot
mcmc_acf(output$mcmc, pars) # autocorrelation