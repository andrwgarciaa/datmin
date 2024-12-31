# Install and load necessary libraries
install.packages(c("tidyverse", "readxl", "factoextra", "cluster", "mclust", "dbscan", "e1071", "kernlab", "fpc"))
library(tidyverse)
library(readxl)
library(factoextra)
library(cluster)
library(mclust)
library(dbscan)
library(e1071)
library(kernlab)
library(fpc)

# Load datasets
retail <- read_excel("data.xlsx")
retail2 <- read_excel("data2.xlsx")

# Summary
summary(retail)
summary(retail2)

# Identify columns with missing values
sapply(retail, function(x) sum(is.na(x)))
sapply(retail2, function(x) sum(is.na(x)))

# Convert InvoiceDate columns to Date type
retail <- retail %>%
  mutate(InvoiceDate = as.Date(InvoiceDate))

# Standardize column names in retail2
# Convert InvoiceDate in retail2 to Date
# Convert UnitPrice in retail2 to numeric
# Adjust InvoiceDate in retail2 to the same range as retail1
retail2 <- retail2 %>%
  rename(UnitPrice = Price, InvoiceDate = TransactionDate) %>%
  mutate(InvoiceDate = as.POSIXct(InvoiceDate, format = "%m/%d/%Y %H:%M")) %>%
  mutate(InvoiceDate = as.Date(InvoiceDate)) %>%
  mutate(UnitPrice = as.numeric(UnitPrice)) %>%
  mutate(InvoiceDate = as.Date(InvoiceDate, format = "%Y-%m-%d") - as.difftime(365 * 13, units = "days"))

# Combine the datasets
retail_combined <- bind_rows(retail, retail2)

# Find the earliest invoice date in the combined dataset
min_invoice_date <- min(as.Date(retail_combined$InvoiceDate, format = "%Y-%m-%d"))

# Data Cleaning
retail_cleaned <- retail_combined %>%
  drop_na(CustomerID) %>%
  filter(Quantity > 0, UnitPrice > 0)

# Add a "Spent" column
retail_cleaned <- mutate(retail_cleaned, Spent = Quantity * UnitPrice)

# Select columns and check missing values
retail_cleaned <- retail_cleaned %>%
  select(CustomerID, Spent, InvoiceDate)

sapply(retail_cleaned, function(x) sum(is.na(x)))

# Check for outliers and skew value
summary(retail_cleaned)
skewness(retail_cleaned$Spent)

# Handling outlier/skew
retail_cleaned$Spent <- log(retail_cleaned$Spent + 1)

# Set a fixed seed for reproducibility
set.seed(123)

# Use the first 50000 rows
retail_cleaned <- retail_cleaned %>%
  slice(1:50000) 

# Filter from outliers
retail_cleaned <- retail_cleaned %>%
  filter(Spent < 10, Spent > 1)

# Summarize customer data
customer <- retail_cleaned %>%
  group_by(CustomerID) %>%
  summarise(
    Spent = sum(Spent, na.rm = TRUE),
)

# RFM Feature Engineering - Recency
recency <- retail_combined %>%
  select(CustomerID, InvoiceDate) %>%
  mutate(recency = as.numeric(as.Date("2011-12-09") - as.Date(InvoiceDate))) %>%
  group_by(CustomerID) %>%
  slice(which.min(as.Date("2011-12-09") - as.Date(InvoiceDate))) %>%
  ungroup() %>%
  mutate(recency = ifelse(recency < 0, 0, recency))  # Replace negative values with 0

# Calculate the number of products purchased in each transaction
amount_products <- retail_cleaned %>%
  group_by(CustomerID, InvoiceDate) %>%
  summarize(n_prod = n(), .groups = "drop")

# Calculate the frequency of transactions
df_frequency <- amount_products %>%
  group_by(CustomerID) %>%
  summarize(frequency = n())

# Extract monetary value
monetary <- select(customer, c("CustomerID", "Spent"))

# Combine RFM Data
rfm <- recency %>%
  inner_join(df_frequency, by = "CustomerID") %>%
  inner_join(monetary, by = "CustomerID") %>%
  select(-CustomerID)

# Scale the RFM features
rfm_norm <- scale(select(rfm, where(is.numeric)))

# Elbow Method for Optimal Clusters
fviz_nbclust(rfm_norm, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2) +
  labs(subtitle = "Elbow Method")

# K-Means Clustering
km.res <- kmeans(rfm_norm, centers = 3, nstart = 50)  # nstart = 50 for better initialization
fviz_cluster(km.res, data = rfm_norm, geom = "point") +
  ggtitle("K-Means Clustering (k = 3)")

# Silhouette for K-Means
km_sil <- silhouette(km.res$cluster[km.res$cluster > 0], dist(rfm_norm[km.res$cluster > 0, ]))
fviz_silhouette(km_sil, title = "DBSCAN Silhouette")

# Add cluster labels to the original RFM data
rfm_with_clusters <- rfm %>%
  mutate(Cluster = as.factor(km.res$cluster))

# View first rows of the RFM data with assigned clusters
head(rfm_with_clusters)

# DBSCAN Clustering with Optimal Parameters
kNNdistplot(rfm_norm, k = 3)
abline(h = 0.5, col = "red", lty = 2)

db <- dbscan(rfm_norm, eps = 0.5) 
fviz_cluster(db, data = rfm_norm,
             geom = "point", palette = "jco",
             main = "DBSCAN Clustering Visualization")

# Silhouette for DBSCAN
db_sil <- silhouette(db$cluster[db$cluster > 0], dist(rfm_norm[db$cluster > 0, ]))
fviz_silhouette(db_sil, title = "DBSCAN Silhouette")

rfm_with_dbscan_clusters <- rfm %>%
  mutate(Cluster = as.factor(db$cluster))

# Exclude noise
rfm_with_dbscan_clusters <- rfm_with_dbscan_clusters %>%
  filter(Cluster != "0")

# View DBSCAN results
head(rfm_with_dbscan_clusters)

# GMM Clustering with 1 to 3 Components
gm <- Mclust(rfm_norm, G = 1:3)
fviz_mclust(gm, what = "classification",
            geom = "point", palette = "jco",
            main = "GMM Clustering Visualization")

# Silhouette for GMM
gm_sil <- silhouette(gm$classification, dist(rfm_norm))
fviz_silhouette(gm_sil, title = "GMM Silhouette")

rfm_with_gmm_clusters <- rfm %>%
  mutate(Cluster = as.factor(gm$classification))

# View GMM results
head(rfm_with_gmm_clusters)

# Hierarchical Clustering
rfm_dist <- dist(rfm_norm, method = "euclidean")
hclust_res <- hclust(rfm_dist, method = "ward.D2")

# Dendrogram visualization
fviz_dend(hclust_res, k = 3,  # k = 3 clusters
          cex = 0.5,          # Label size
          rect = TRUE,        # Add rectangles around clusters
          main = "Hierarchical Clustering Dendrogram")

# Cut tree into k clusters
hc_clusters <- cutree(hclust_res, k = 3)

# Visualize Hierarchical Clustering
fviz_cluster(list(data = rfm_norm, cluster = hc_clusters),
             geom = "point", ellipse = TRUE,
             main = "Hierarchical Clustering (k = 3)")

# Silhouette Analysis for Hierarchical Clustering
hc_sil <- silhouette(hc_clusters, rfm_dist)
fviz_silhouette(hc_sil, title = "Hierarchical Clustering Silhouette")

# Add cluster labels to the original RFM data
rfm_with_hclust_clusters <- rfm %>%
  mutate(Cluster = as.factor(hc_clusters))

# View results
head(rfm_with_hclust_clusters)

kmeans_cluster_summary <- rfm_with_clusters %>%
  group_by(Cluster) %>%
  summarize(
    Avg_Recency = mean(recency),
    Avg_Frequency = mean(frequency),
    Avg_Monetary = mean(Spent),
    Count = n()
  )

print(kmeans_cluster_summary)

dbscan_cluster_summary <- rfm_with_dbscan_clusters %>%
  group_by(Cluster) %>%
  summarize(
    Avg_Recency = mean(recency),
    Avg_Frequency = mean(frequency),
    Avg_Monetary = mean(Spent),
    Count = n()
  )

print(dbscan_cluster_summary)

gmm_cluster_summary <- rfm_with_gmm_clusters %>%
  group_by(Cluster) %>%
  summarize(
    Avg_Recency = mean(recency),
    Avg_Frequency = mean(frequency),
    Avg_Monetary = mean(Spent),
    Count = n()
  )

print(gmm_cluster_summary)

hierarchical_cluster_summary <- rfm_with_hclust_clusters %>%
  group_by(Cluster) %>%
  summarize(
    Avg_Recency = mean(recency),
    Avg_Frequency = mean(frequency),
    Avg_Monetary = mean(Spent),
    Count = n()
  )

print(hierarchical_cluster_summary)

# Silhouette Score, Dunn, and Calinski-Harabasz indices for K-Means
km_stats <- cluster.stats(dist(rfm_norm), km.res$cluster)
dunn_kmeans <- km_stats$dunn
ch_kmeans <- km_stats$ch
silhouette_score_kmeans <- km_stats$avg.silwidth 

# Silhouette Score, Dunn, and Calinski-Harabasz indices for DBSCAN
db_stats <- cluster.stats(dist(rfm_norm), db$cluster)
dunn_dbscan <- db_stats$dunn
ch_dbscan <- db_stats$ch
silhouette_score_dbscan <- db_stats$avg.silwidth  

# Silhouette Score, Dunn, and Calinski-Harabasz indices for GMM
gmm_stats <- cluster.stats(dist(rfm_norm), gm$classification)
dunn_gmm <- gmm_stats$dunn
ch_gmm <- gmm_stats$ch
silhouette_score_gmm <- gmm_stats$avg.silwidth 

# Silhouette Score Dunn, and Calinski-Harabasz indices for Hierarchical Clustering
hclust_stats <- cluster.stats(dist(rfm_norm), hc_clusters)
dunn_hclust <- hclust_stats$dunn
ch_hclust <- hclust_stats$ch
silhouette_score_hclust <- hclust_stats$avg.silwidth


# Create a data frame to summarize all evaluation metrics
evaluation_metrics <- data.frame(
  method = c("K-Means", "DBSCAN", "GMM", "Hierarchical"),
  silhouette_score = c(silhouette_score_kmeans, silhouette_score_dbscan, silhouette_score_gmm, silhouette_score_hclust),
  dunn_index = c(dunn_kmeans, dunn_dbscan, dunn_gmm, dunn_hclust),
  calinski_harabasz = c(ch_kmeans, ch_dbscan, ch_gmm, ch_hclust)
)

print(evaluation_metrics)

