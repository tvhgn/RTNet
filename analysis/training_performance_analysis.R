library(ggplot2)
library(dplyr)
library(tidyr)
library(viridis)

# Read the data
df <- read.csv("results/training_performance_RTNet.csv")

# Plot validation loss and accuracy against epoch
epochs <- 1:30

# convert data to long format
df <- df %>% 
  mutate(X=X+1) %>%
  mutate(TrainingLoss=log(TrainingLoss)) %>%
  mutate(ValLoss=log(ValLoss)) %>%
  pivot_longer(cols=c("TrainingLoss","ValLoss", "ValAcc"),
                          names_to="ValMetric",
                          values_to="value")


# Plot the data
colors = viridis(10)

ggplot(data=df, aes(x=X, y=value)) +
  geom_line(aes(color=ValMetric), size=0.9) +
  scale_color_manual(values=c(
    "TrainingLoss" = colors[3],
    "ValLoss" = colors[7],
    "ValAcc" = colors[8]
  ), name=NULL, labels=c("Training Loss", "Validation Accuracy", "Validation Loss")) + 
  xlab("Epoch") +
  ylab("Log(Loss)") +
  labs(tag="% Accurate") + 
  theme(legend.box.margin=margin(l=20),
    plot.tag = element_text(angle=-90, size=11),
    plot.tag.position=c(.73, 0.55))

