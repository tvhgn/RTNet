library(dplyr)
library(papaja)
library(ggplot2)
library(gridExtra)

# Load simulations datafile
df <- read.csv(file.path("results", "simulations.csv"))

# Get distributions grouped on threshold level and noise level
df_p <- df %>%
  mutate(true.label = factor(true.label)) %>%
  mutate(choice = factor(choice))

# Create RT distributions for each group (threshold x noise)
thresholds <- unique(df$threshold)
noises <- unique(df$noise)

# Create a list to store plots
plot_list <- list()

# Iterate over threshold levels, then noise levels
for (t in thresholds) {
  for (n in noises) {
    # Filter dataframe
    filtered_df <- df_p %>%
      filter(threshold == t & noise == n)
    
    # Create histogram and density plot
    p <- ggplot(data = filtered_df, aes(x = rt)) +
      geom_histogram(aes(y = ..density..), binwidth=1,fill = 'blue', color = 'black', alpha=0.5) +
      geom_density(color='red', fill='red', alpha=0.5, bw=0.5) +
      labs(title = paste("threshold =", t, ", noise =", n), x = "RT", y = "Density") +
      xlim(0,25) +
      ylim(0, 0.45)
      theme_apa()
    
    # Store the plot in the list
    plot_list[[paste(t, n, sep = "_")]] <- p
  }
}

# Arrange and display all plots in a grid
do.call(grid.arrange, c(plot_list, ncol = 2))
