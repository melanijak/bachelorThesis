### Content

#### Data processing
* process_data.py 
Reads the downloaded data from FI and filters the data. Creates two data frames, one containing the baseline visits, and the other containing the remaining visits. 
* normalize_values
Normalizes the values of the data
* make_views
Divides the data into separate views, to be used later in multi-view clustering

#### Clustering
* single_view_kmeans.py
Performs NMF and kmeans clustering.
* multi_view_spectral.py
Performs multi veiw spectral clustering
* stats.py
Performs the Kolmogorov-Smirnov statistic on pairs of cluster labels

#### Rules
* cluster_rules
Trains a decision tree models with a data set containing baseline visits and cluster assignments. Prints rules

#### Progression
* predict_clusters.py
Trains a RandomForest model with data set containing baseline visits and cluster assignments. Predicts the cluster labels on the remaining visits.
* concatinate_views.py
Creates a table containing rows with patient ID and sequence of cluster assignments of all visits.
* skipgrams.py
Performs skipgrams and creates histogram.
