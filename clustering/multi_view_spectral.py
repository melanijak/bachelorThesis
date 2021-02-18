from multiview.mvsc import MVSC
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

# computes SA, DB and CH index scores
def compute_scores(X, cluster_labels):

    sa = silhouette_score(X, cluster_labels)
    ch = calinski_harabasz_score(X, cluster_labels)
    db = davies_bouldin_score(X, cluster_labels)
    return [round(sa, 4), round(ch, 4), round(db, 4)]

# read all views and remove the ID column
dir = '../views/MVC/'
df1 = pd.read_csv(dir +'PD_CognitionDaily.csv')
df1 = df1.pop('fox_insight_id')
df2 = pd.read_csv(dir + 'PD_NonMovement.csv')
df2 = df2.pop('fox_insight_id')
df3 = pd.read_csv(dir + 'PD_Movement.csv')
d3 = df3.pop('fox_insight_id')
df4 = pd.read_csv(dir + 'PD_PhysicalExperience.csv')
df4 = df4.pop('fox_insight_id')
df5 = pd.read_csv(dir + 'PD_PhysicalActivities.csv')
df5 = df5.pop('fox_insight_id')
df6 = pd.read_csv(dir + 'PD_MedsOther.csv')
df6 = df6.pop('fox_insight_id')
df7 = pd.read_csv(dir + 'PD_MedsPDCurrent.csv')
df7.pop('fox_insight_id')

df = pd.read_csv('../NMF/fox_data_original.csv')
dX = df.drop('fox_insight_id', 1)

# define number of desired clusters
n_clusters = 2

# define multi-view spectral clustering model
mvsc = MVSC(k=n_clusters)
# input views
clust = mvsc.fit_transform([df1.values, df2.values, df3.values, df4.values, df5.values, df6.values, df7.values],  [False] * 7)
# read scores from clustering
clustering = clust[0]


scores = []
# compute indexes for all views (just for informative purposes)
scores.append(compute_scores(df1, clustering))
scores.append(compute_scores(df2, clustering))
scores.append(compute_scores(df3, clustering))
scores.append(compute_scores(df4, clustering))
scores.append(compute_scores(df5, clustering))
scores.append(compute_scores(df6, clustering))
scores.append(compute_scores(df7, clustering))
scores = np.array(scores)
scores = np.round(scores, decimals=4)
print(scores)
print(scores.mean(axis=0))

print('Overall')
# compute scores on the original matrix with merged views
print(compute_scores(dX, clustering))
df['Cluster'] = clustering
print(df.head())
#df.to_csv('BL_10do15_multi_view_clusters.csv')