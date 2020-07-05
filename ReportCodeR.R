library(randomForest)
library(caret)
library(cluster)
library(ggplot2)
library(factoextra)
library(NbClust)


#split

cluster_data = read.csv('C:/Users/laura/Data_prelim/cluster_data.csv')
#rf_split <-  randomForest(x = cluster_data['chlfa'], mtry = 1, ntree = 1000, proximity = TRUE)
#prox_split = rf_split$proximity
#write.csv(prox_split,"C:/Users/laura/Data_prelim/random_forest_split.csv", row.names = FALSE)


#offshore
offshore = read.csv('C:/Users/laura/Data_prelim/df_offshore.csv')
#rf_offshore <-  randomForest(x = offshore, mtry = 2, ntree = 1000, proximity = TRUE)
#prox_offshore = rf_offshore$proximity
#write.csv(prox_offshore,"C:/Users/laura/Data_prelim/random_forest_offshore", row.names = FALSE)

#coast
coast = read.csv('C:/Users/laura/Data_prelim/coast.csv')
#rf_coast <-  randomForest(x = coast, mtry = 2, ntree = 1000, proximity = TRUE)
#prox_coast = rf_coast$proximity
#write.csv(prox_coast,"C:/Users/laura/Data_prelim/random_forest_coast", row.names = FALSE)

prox_split = read.csv('C:/Users/laura/Data_prelim/random_forest_split')
prox_offshore = read.csv('C:/Users/laura/Data_prelim/random forest_offshore_prox')
prox_coast = read.csv('C:/Users/laura/Data_prelim/random_forest_coast')




#metrics

hcut_average <- function(x, k) {
  return(hcut(x, k=k, hc_method = "average"))
}

hcut_ward <- function(x, k) {
  return(hcut(x, k=k, hc_method = "ward.D2"))
}

kmeans_offshore_wss <- fviz_nbclust(offshore, kmeans, method = "wss")
kmeans_offshore_sil <- fviz_nbclust(offshore, kmeans, method = "silhouette")
kmeans_offshore_gap <- fviz_nbclust(offshore, kmeans, method = "gap_stat", nboot = 50)

hcut_offshore_wss <- fviz_nbclust(offshore, hcut_ward, method = "wss")
hcut_offshore_sil <- fviz_nbclust(offshore, hcut_ward, method = "silhouette")
hcut_offshore_gap <- fviz_nbclust(offshore, hcut_ward, method = "gap_stat", nboot = 50)


RF_offshore_wss <- fviz_nbclust(offshore, hcut_average, diss = 1-prox_offshore, method = "wss")
RF_offshore_sil <- fviz_nbclust(offshore, hcut_average, diss = 1-prox_offshore, method = "silhouette")
RF_offshore_gap <- fviz_nbclust(offshore, hcut_average, diss = 1-prox_offshore, method = "gap_stat", nboot = 50)


kmeans_split_wss <- fviz_nbclust(cluster_data['chlfa'], kmeans, method = "wss")
kmeans_split_sil <- fviz_nbclust(cluster_data['chlfa'], kmeans, method = "silhouette")
kmeans_split_gap <- fviz_nbclust(cluster_data['chlfa'], kmeans, method = "gap_stat", nboot = 50)

hcut_split_wss <- fviz_nbclust(cluster_data['chlfa'], hcut_ward, method = "wss")
hcut_split_sil <- fviz_nbclust(cluster_data['chlfa'], hcut_ward, method = "silhouette")
hcut_split_gap <- fviz_nbclust(cluster_data['chlfa'], hcut_ward, method = "gap_stat", nboot = 50)


RF_split_wss <- fviz_nbclust(cluster_data['chlfa'], hcut_average, diss = 1-prox_split, method = "wss")
RF_split_sil <- fviz_nbclust(cluster_data['chlfa'], hcut_average, diss = 1-prox_split, method = "silhouette")
RF_split_gap <- fviz_nbclust(cluster_data['chlfa'], hcut_average, diss = 1-prox_split, method = "gap_stat", nboot = 50)


kmeans_coast_wss <- fviz_nbclust(coast, kmeans, method = "wss")
kmeans_coast_sil <- fviz_nbclust(coast, kmeans, method = "silhouette")
kmeans_coast_gap <- fviz_nbclust(coast, kmeans, method = "gap_stat", nboot = 50)

hcut_coast_wss <- fviz_nbclust(coast, hcut_ward, method = "wss")
hcut_coast_sil <- fviz_nbclust(coast, hcut_ward, method = "silhouette")
hcut_coast_gap <- fviz_nbclust(coast, hcut_ward, method = "gap_stat", nboot = 50)


RF_coast_wss <- fviz_nbclust(coast, hcut_average, diss = 1-prox_coast, method = "wss")
RF_coast_sil <- fviz_nbclust(coast, hcut_average, diss = 1-prox_coast, method = "silhouette")
RF_coast_gap <- fviz_nbclust(coast, hcut_average, diss = 1-prox_coast, method = "gap_stat", nboot = 50)

metrics = data.frame(kmeans_offshore_wss$data$y, kmeans_offshore_sil$data$y, kmeans_offshore_gap$data$gap, 
                     hcut_offshore_wss$data$y, hcut_offshore_sil$data$y, hcut_offshore_gap$data$gap,
                     RF_offshore_wss$data$y, RF_offshore_sil$data$y, RF_offshore_gap$data$gap,
                     
                     kmeans_split_wss$data$y, kmeans_split_sil$data$y, kmeans_split_gap$data$gap, 
                     hcut_split_wss$data$y, hcut_split_sil$data$y, hcut_split_gap$data$gap,
                     RF_split_wss$data$y, RF_split_sil$data$y, RF_split_gap$data$gap,
                     
                     kmeans_coast_wss$data$y, kmeans_coast_sil$data$y, kmeans_coast_gap$data$gap, 
                     hcut_coast_wss$data$y, hcut_coast_sil$data$y, hcut_coast_gap$data$gap,
                     RF_coast_wss$data$y, RF_coast_sil$data$y, RF_coast_gap$data$gap)

write.csv(metrics,"C:/Users/laura/Data_prelim/metrics.csv", row.names = FALSE)




kmeans_offshore_wss +
  labs(subtitle = "Elbow method: K-means clustering on 22-03-2003")
kmeans_offshore_sil  +
  labs(subtitle = "Silhouette: K-means clustering on 22-03-2003")
kmeans_offshore_gap +
  labs(subtitle = "Gap statistic: K-means clustering on 22-03-2003")

hcut_offshore_wss +
  labs(subtitle = "Elbow method: Hierarchical clustering 22-03-2003")
hcut_offshore_sil  +
  labs(subtitle = "Silhouette: Hierarchical clustering on 22-03-2003")
hcut_offshore_gap +
  labs(subtitle = "Gap statistic: Hierarchical clustering on 22-03-2003")

RF_offshore_wss +
  labs(subtitle = "Elbow method: Random Forest clustering on 22-03-2003")
RF_offshore_sil  +
  labs(subtitle = "Silhouette: Random Forest clustering on 22-03-2003")
RF_offshore_gap +
  labs(subtitle = "Gap statistic: Random Forest clustering on 22-03-2003")





kmeans_split_wss +
  labs(subtitle = "Elbow method: K-means clustering on 17-02-2003")
kmeans_split_sil  +
  labs(subtitle = "Silhouette: K-means clustering on 17-02-2003")
kmeans_split_gap +
  labs(subtitle = "Gap statistic: K-means clustering on 17-02-2003")


hcut_split_wss +
  labs(subtitle = "Elbow method: Hierarchical clustering on 17-02-2003")
hcut_split_sil  +
  labs(subtitle = "Silhouette: Hierarchical clustering on 17-02-2003")
hcut_split_gap +
  labs(subtitle = "Gap statistic: Hierarchical clustering on 17-02-2003")

RF_split_wss +
  labs(subtitle = "Elbow method: Random Forest clustering on 17-02-2003")
RF_split_sil  +
  labs(subtitle = "Silhouette: Random Forest clustering on 17-02-2003")
RF_split_gap +
  labs(subtitle = "Gap statistic: Random Forest clustering on 17-02-2003")




kmeans_coast_wss +
  labs(subtitle = "Elbow method: K-means clustering on 25-02-2003")
kmeans_coast_sil  +
  labs(subtitle = "Silhouette: K-means clustering on 25-02-2003")
kmeans_coast_gap +
  labs(subtitle = "Gap statistic: K-means clustering on 25-02-2003")

hcut_coast_wss +
  labs(subtitle = "Elbow method: Hierarchical clustering on 25-02-2003")
hcut_coast_sil  +
  labs(subtitle = "Silhouette: Hierarchical clustering on 25-02-2003")
hcut_coast_gap +
  labs(subtitle = "Gap statistic: Hierarchical clustering on 25-02-2003")

RF_coast_wss +
  labs(subtitle = "Elbow method: Random Forest clustering on 25-02-2003")
RF_coast_sil  +
  labs(subtitle = "Silhouette: Random Forest clustering on 25-02-2003")
RF_coast_gap +
  labs(subtitle = "Gap statistic: Random Forest clustering on 25-02-2003")


