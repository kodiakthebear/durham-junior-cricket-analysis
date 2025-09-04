library(tidyverse)
library(ggplot2)
library(factoextra)
library(corrplot)
library(ggrepel)

#Read data
data <- read.csv("/Users/mukundranjan/Documents/Academics/Easter/Dissertation/Diss Code and DBs/Databases/merged data/cdata.csv")

# Data preprocessing
#Remove non-numeric columns and handle missing values
pca_data <- data %>%
  select(-Player, -SELECTION) %>%  # Remove player names and target variable
  select_if(is.numeric) %>%        # Keep only numeric columns
  mutate_all(~ifelse(is.na(.), 0, .)) %>%  # Replace NA with 0
  # Remove columns with zero variance
  select_if(~var(., na.rm = TRUE) != 0)

#Scale the data 
pca_scaled <- scale(pca_data)

#Perform PCA
pca_result <- prcomp(pca_scaled, center = FALSE, scale. = FALSE)

#Create a comprehensive PCA visualization function
create_enhanced_pca_plot <- function(pca_result, original_data, title_suffix = "") {
  
  # Extract PC scores
  pca_scores <- as.data.frame(pca_result$x)
  pca_scores$Player <- original_data$Player
  pca_scores$Selection <- factor(original_data$SELECTION, 
                                 levels = c(0, 1), 
                                 labels = c("Not Selected", "Selected"))
  
  # Calculate variance explained
  var_explained <- summary(pca_result)$importance[2, 1:2] * 100
  
  # Create the main PCA plot with better aesthetics
  p1 <- ggplot(pca_scores, aes(x = PC1, y = PC2)) +
    geom_point(aes(color = Selection, shape = Selection), 
               size = 3, alpha = 0.8) +
    scale_color_manual(values = c("Not Selected" = "#440154", 
                                  "Selected" = "#FDE725")) +
    scale_shape_manual(values = c("Not Selected" = 16, 
                                  "Selected" = 17)) +
    labs(
      title = paste("PCA: Cricket Player Performance Analysis", title_suffix),
      subtitle = paste("Players colored by selection status"),
      x = paste0("PC1 (", round(var_explained[1], 1), "% of variance)"),
      y = paste0("PC2 (", round(var_explained[2], 1), "% of variance)"),
      color = "Selection Status",
      shape = "Selection Status"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      axis.title = element_text(size = 11)
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.5)
  
  return(list(plot = p1, scores = pca_scores, var_explained = var_explained))
}

# Create the enhanced plot
pca_enhanced <- create_enhanced_pca_plot(pca_result, data)
print(pca_enhanced$plot)


# Create a biplot showing variable contributions
fviz_pca_biplot(pca_result, 
                geom.ind = c("point"),
                col.ind = factor(data$SELECTION, labels = c("Not Selected", "Selected")),
                palette = c("#440154", "#FDE725"),
                addEllipses = TRUE,
                ellipse.level = 0.68,
                repel = TRUE,
                title = "PCA Biplot: Players and Variable Loadings",
                subtitle = "Arrows show variable contributions to PCs")

# Show the most important variables for each PC
loadings_df <- as.data.frame(pca_result$rotation[, 1:2])
loadings_df$Variable <- rownames(loadings_df)

# Top contributors to PC1
pc1_contrib <- loadings_df %>%
  arrange(desc(abs(PC1))) %>%
  head(10)

# Top contributors to PC2  
pc2_contrib <- loadings_df %>%
  arrange(desc(abs(PC2))) %>%
  head(10)

cat("Top 10 variables contributing to PC1:\n")
print(pc1_contrib[c("Variable", "PC1")])

cat("\nTop 10 variables contributing to PC2:\n") 
print(pc2_contrib[c("Variable", "PC2")])

# Create a plot showing variable contributions
contrib_plot <- loadings_df %>%
  pivot_longer(cols = c(PC1, PC2), names_to = "Component", values_to = "Loading") %>%
  group_by(Component) %>%
  slice_max(abs(Loading), n = 8) %>%
  ggplot(aes(x = reorder(Variable, abs(Loading)), y = Loading, fill = Component)) +
  geom_col(alpha = 0.8) +
  coord_flip() +
  facet_wrap(~Component, scales = "free_y") +
  labs(title = "Variable Contributions to Principal Components",
       subtitle = "Top 8 contributing variables for each PC",
       x = "Variables", y = "Loading") +
  theme_minimal() +
  theme(legend.position = "none")

print(contrib_plot)

# Scree plot
variance_plot <- fviz_eig(pca_result, 
                          addlabels = TRUE, 
                          ylim = c(0, 50),
                          main = "Variance Explained by Principal Components",
                          subtitle = "Scree plot showing contribution of each PC")
print(variance_plot)

# Summary statistics by group
summary_stats <- pca_enhanced$scores %>%
  group_by(Selection) %>%
  summarise(
    n = n(),
    PC1_mean = round(mean(PC1), 2),
    PC1_sd = round(sd(PC1), 2),
    PC2_mean = round(mean(PC2), 2),
    PC2_sd = round(sd(PC2), 2),
    .groups = 'drop'
  )

cat("\nSummary statistics by selection status:\n")
print(summary_stats)

# Add player labels for extreme cases
extreme_players <- pca_enhanced$scores %>%
  filter(abs(PC1) > 2*sd(PC1) | abs(PC2) > 2*sd(PC2))

if(nrow(extreme_players) > 0) {
  labeled_plot <- pca_enhanced$plot +
    geom_text_repel(data = extreme_players, 
                    aes(label = Player, color = Selection),
                    size = 3,
                    show.legend = FALSE,
                    max.overlaps = 10)
  
  print(labeled_plot)
}