import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans

# read file with normalized and merged attributes
df = pd.read_csv('fox_data_baseline_10do15.csv')
X = df.drop('fox_insight_id', 1)

# info about data
print("Number of patients: ", len(df))
print("Number of symptoms/attributes: ", len(df.columns))

# perform NMF on data
from sklearn.decomposition import NMF
maxiter = 1000
components = 2
model = NMF(n_components=components, max_iter=maxiter)
model.fit(X)
nmf_features = model.transform(X)
print(nmf_features.shape)
# the X has now reduces attributes (in this case 2)
X = nmf_features

# method that computes normalized version of SA
def get_norm_silhouette(indices, sil_score, sil_samples):
    ratios = []
    for cls in indices:
        inds = indices[cls]
        t_inds = []
        for ti in inds:
            if sil_samples[ti] - sil_score >= 1e-10:
                t_inds.append(ti)
        ratio = (1.0 * len(t_inds) / len(inds)) * (1.0 * len(inds) / len(sil_samples))
        ratios.append(ratio)

    return sum(ratios) * sil_score

def evaluate_clustering(data, clustering):
    sil_score = silhouette_score(data, clustering)
    sil_samples = silhouette_samples(data,clustering)
    indices = get_cluster_indices(clustering)
    norm_sil_score = get_norm_silhouette(indices, sil_score, sil_samples)
    return norm_sil_score


def get_cluster_indices(cls):
    indices = dict()
    for i in range(len(cls)):
        if cls[i] not in indices:
            indices[cls[i]] = []
        l = indices[cls[i]]
        l.append(i)
        indices[cls[i]] = l
    return indices

# method that computes SA, SA normal, CH and DB index scores
def compute_scores(X, cluster_labels):
    sa = silhouette_score(X, cluster_labels)
    sa_normal = evaluate_clustering(X, cluster_labels)
    ch = calinski_harabasz_score(X, cluster_labels)
    db = davies_bouldin_score(X, cluster_labels)
    return [round(sa,4), round(sa_normal,4), round(ch,4), round(db,4)]

# for each index prints the k with the best score
def optimal_k(scores, low, high):
    print(scores)
    optimal_k = np.argmax(scores, axis=0)
    optimal_k_min = np.argmin(scores, axis=0)
    optimal_k = [x+low for x in optimal_k]
    s_names = ["SA", "SA normal", "CH", "DB"]

    for i in range(len(s_names)):
        if s_names[i] == "DB":
            print("k: ", optimal_k_min[i], " ", s_names[i], ": ", round(scores[optimal_k_min[i] - low][i], 3))
        else:
            print("k: ", optimal_k[i], " ", s_names[i], ": ", round(scores[optimal_k[i]-low][i],3))


# perform clustering on different number of desired clusters (from 2 to 7)
low=2
high=7

# K means clustering
kmeans_scores = []
for n_clusters in range(low, high):
     clusterer = KMeans(n_clusters=n_clusters, random_state=42)
     cluster_labels = clusterer.fit_predict(X)
     print("Cluster labels")
     for label in cluster_labels:
         print(label, end =" ")

     ime = 'noCluster' + str(n_clusters)
     df[ime] = cluster_labels
     kmeans_scores.append(compute_scores(X, cluster_labels))

print("**K-MEANS***")
optimal_k(kmeans_scores, low, high)

# when the optimal k is determined, save a new csv file with the cluster label
labels = dict()
final_k = 3
clusterer = KMeans(n_clusters=final_k, random_state=10)
cluster_labels = clusterer.fit_predict(X)


cluster_labels = [[x] for x in cluster_labels]
clusters_df = np.append(X, cluster_labels, axis = 1)
columns = df.columns.tolist() + ['Cluster']
columns.pop(0)
print(columns)

clusters_df = pd.DataFrame(clusters_df, columns=columns)
clusters_df.to_csv('fox_data_clusters_k3.csv')