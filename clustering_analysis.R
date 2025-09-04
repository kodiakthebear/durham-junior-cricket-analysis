library(tidyverse)
library(ggplot2)
library(factoextra)
library(cluster)
library(gridExtra)
library(corrplot)
library(ggrepel)

# Read data
data <- read.csv("/Users/mukundranjan/Documents/Academics/Easter/Dissertation/Diss Code and DBs/Databases/merged data/cdata.csv")

# Data preprocessing (same as PCA)
cluster_data <- data %>%
  select(-Player, -SELECTION) %>%  # Remove player names and target variable
  select_if(is.numeric) %>%        # Keep only numeric columns
  mutate_all(~ifelse(is.na(.), 0, .)) %>%  # Replace NA with 0
  # Remove columns with zero variance
  select_if(~var(., na.rm = TRUE) != 0)

# Scale the data
cluster_scaled <- scale(cluster_data)

# STEP 1 - DETERMINE OPTIMAL NUMBER OF CLUSTERS
# Method 1: Elbow method
set.seed(123)  # For reproducibility
wss <- fviz_nbclust(cluster_scaled, kmeans, method = "wss", k.max = 10) +
  labs(title = "Elbow Method: Optimal Number of Clusters",
       subtitle = "Looking for the 'elbow' point where WSS stops decreasing rapidly")

# Method 2: Silhouette method
silhouette <- fviz_nbclust(cluster_scaled, kmeans, method = "silhouette", k.max = 10) +
  labs(title = "Silhouette Method: Optimal Number of Clusters",
       subtitle = "Higher silhouette score indicates better clustering")

# Method 3: Gap statistic method
set.seed(123)
gap_stat <- fviz_nbclust(cluster_scaled, kmeans, method = "gap_stat", k.max = 10) +
  labs(title = "Gap Statistic Method: Optimal Number of Clusters",
       subtitle = "K where gap statistic is maximized")

# Display all three methods
grid.arrange(wss, silhouette, gap_stat, ncol = 1)

# STEP 2 - PERFORM K-MEANS CLUSTERING
optimal_k <- 4 #Optimal k chosen after trial and error

set.seed(123)
kmeans_result <- kmeans(cluster_scaled, centers = optimal_k, nstart = 25)

# Add cluster assignments to original data
data_clustered <- data %>%
  mutate(Cluster = factor(kmeans_result$cluster),
         Selection = factor(SELECTION, levels = c(0, 1), labels = c("Not Selected", "Selected")))

# STEP 3 - VISUALIZE CLUSTERS IN PCA SPACE
# Perform PCA for visualization
pca_result <- prcomp(cluster_scaled, center = FALSE, scale. = FALSE)
pca_scores <- as.data.frame(pca_result$x[, 1:2])
pca_scores$Cluster <- factor(kmeans_result$cluster)
pca_scores$Player <- data$Player
pca_scores$Selection <- data_clustered$Selection

# Calculate variance explained
var_explained <- summary(pca_result)$importance[2, 1:2] * 100

# Create PCA plot colored by clusters
cluster_pca_plot <- ggplot(pca_scores, aes(x = PC1, y = PC2)) +
  geom_point(aes(color = Cluster, shape = Selection), size = 3, alpha = 0.8) +
  scale_color_brewer(type = "qual", palette = "Set1") +
  scale_shape_manual(values = c("Not Selected" = 16, "Selected" = 17)) +
  labs(
    title = paste("K-Means Clusters (k =", optimal_k, ") in PCA Space"),
    subtitle = "Clusters colored, selection status shown by shape",
    x = paste0("PC1 (", round(var_explained[1], 1), "% of variance)"),
    y = paste0("PC2 (", round(var_explained[2], 1), "% of variance)")
  ) +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        legend.position = "bottom") +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.5)

print(cluster_pca_plot)

# STEP 4 - CLUSTER CHARACTERIZATION
# Calculate cluster centres in original scale
cluster_centers <- data_clustered %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = 'drop') %>%
  select(-SELECTION)

# Key cricket statistics for interpretation
key_stats <- c("AVG", "STRIKE.RATE", "WICKETS", "ECONOMY.RATE", 
               "TOTAL.VICTIMS", "BOWL.STRIKE.RATE", "BOWL.AVERAGE")

# Create heatmap of cluster characteristics
cluster_summary <- cluster_centers %>%
  select(Cluster, all_of(key_stats[key_stats %in% names(cluster_centers)])) %>%
  column_to_rownames("Cluster") %>%
  as.matrix()

#Scale for heatmap visualization
cluster_summary_scaled <- scale(cluster_summary)

# Create heatmap
cluster_heatmap_data <- cluster_summary_scaled %>%
  as.data.frame() %>%
  rownames_to_column("Cluster") %>%
  pivot_longer(-Cluster, names_to = "Statistic", values_to = "Value")

heatmap_plot <- ggplot(cluster_heatmap_data, aes(x = Statistic, y = Cluster, fill = Value)) +
  geom_tile() +
  scale_fill_gradient2(low = "#440154", mid = "white", high = "#FDE725", 
                       name = "Standardized\nValue") +
  labs(title = "Cluster Characteristics Heatmap",
       subtitle = "Standardized values of key cricket statistics",
       x = "Cricket Statistics", y = "Cluster") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 14, face = "bold"))

print(heatmap_plot)

# STEP 5 - DETAILED CLUSTER ANALYSIS
# Selection rate by cluster
selection_by_cluster <- data_clustered %>%
  group_by(Cluster) %>%
  summarise(
    n_players = n(),
    selected = sum(SELECTION),
    selection_rate = round(selected / n_players * 100, 1),
    .groups = 'drop'
  )

print("Selection rates by cluster:")
print(selection_by_cluster)

# Statistical summary by cluster
cluster_stats <- data_clustered %>%
  group_by(Cluster) %>%
  summarise(
    Players = n(),
    Avg_Games = round(mean(GAMES, na.rm = TRUE), 1),
    Avg_Batting_Avg = round(mean(AVG, na.rm = TRUE), 1),
    Avg_Strike_Rate = round(mean(STRIKE.RATE, na.rm = TRUE), 1),
    Avg_Wickets = round(mean(WICKETS, na.rm = TRUE), 1),
    Avg_Economy = round(mean(ECONOMY.RATE, na.rm = TRUE), 1),
    Selection_Rate = paste0(round(sum(SELECTION)/n()*100, 1), "%"),
    .groups = 'drop'
  )

print("Cluster statistics summary:")
print(cluster_stats)

# STEP 6 - CLUSTER PROFILING
# Create radar chart for each cluster
create_radar_data <- function(cluster_num, data_clustered) {
  cluster_data <- data_clustered %>% 
    filter(Cluster == cluster_num) %>%
    select(AVG, STRIKE.RATE, WICKETS, ECONOMY.RATE, TOTAL.CATCHES, GAMES) %>%
    summarise_all(mean, na.rm = TRUE)
  
  # Normalize to 0-1 scale for radar chart
  normalized <- as.data.frame(lapply(cluster_data, function(x) {
    if(max(data_clustered[[names(cluster_data)[which(cluster_data == x)]]], na.rm = TRUE) == 0) return(0)
    x / max(data_clustered[[names(cluster_data)[which(cluster_data == x)]]], na.rm = TRUE)
  }))
  
  return(normalized)
}

# STEP 7 - IDENTIFY REPRESENTATIVE PLAYERS BY CLUSTER
# Find players closest to cluster centers
find_representative_players <- function(cluster_num, data_clustered, cluster_scaled) {
  cluster_center <- kmeans_result$centers[cluster_num, ]
  cluster_players <- which(kmeans_result$cluster == cluster_num)
  
  # Calculate distances to cluster center
  distances <- apply(cluster_scaled[cluster_players, ], 1, function(x) {
    sqrt(sum((x - cluster_center)^2))
  })
  
  # Get the 3 closest players
  closest_indices <- cluster_players[order(distances)[1:min(3, length(distances))]]
  
  return(data$Player[closest_indices])
}

# Get representative players for each cluster
cat("Representative players for each cluster:\n")
for(i in 1:optimal_k) {
  cat(paste("Cluster", i, ":\n"))
  representatives <- find_representative_players(i, data_clustered, cluster_scaled)
  cat(paste("-", representatives, collapse = "\n"))
  cat("\n\n")
}

# STEP 8 - CLUSTER VALIDATION
# Silhouette analysis
sil <- silhouette(kmeans_result$cluster, dist(cluster_scaled))
fviz_silhouette(sil) +
  labs(title = "Silhouette Analysis of K-Means Clustering",
       subtitle = paste("Average silhouette width:", round(mean(sil[, 3]), 2)))

# STEP 9 - COMPARE WITH ACTUAL SELECTION
# Cross-tabulation of clusters vs selection
cluster_selection_table <- table(data_clustered$Cluster, data_clustered$Selection)
print("Cluster vs Selection Cross-tabulation:")
print(cluster_selection_table)

# Chi-square test for independence
chi_test <- chisq.test(cluster_selection_table)
cat(paste("\nChi-square test p-value:", round(chi_test$p.value, 4)))
if(chi_test$p.value < 0.05) {
  cat("\nClusters are significantly associated with selection status!")
} else {
  cat("\nNo significant association between clusters and selection status.")
}

# STEP 10 - ACTIONABLE INSIGHTS
# Identify potentially undervalued players
undervalued_players <- data_clustered %>%
  group_by(Cluster) %>%
  mutate(cluster_selection_rate = mean(SELECTION)) %>%
  ungroup() %>%
  filter(SELECTION == 0, cluster_selection_rate > 0.5) %>%  # Not selected but in high-selection cluster
  select(Player, Cluster, cluster_selection_rate, AVG, STRIKE.RATE, WICKETS)

if(nrow(undervalued_players) > 0) {
  cat("\nPotentially undervalued players (not selected but in high-selection clusters):\n")
  print(undervalued_players)
} else {
  cat("\nNo obviously undervalued players identified.")
}