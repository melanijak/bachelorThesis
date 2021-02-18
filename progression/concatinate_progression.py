import pandas as pd

# read file containing baseline visits and their cluster labels
df1 = pd.read_csv('fox_data_clusters_k3.csv')
# read file containing remaining visits and their predicted cluster labels
df2 = pd.read_csv('fox_data_clusters_k3_prediction.csv', index_col=0)
cluster_column = 'Cluster'
cols = ['fox_insight_id', cluster_column]
df1 = df1[cols]
df2 = df2[cols]
# concatinade only the index and the cluster label for all visits
df = pd.concat([df1, df2]).reset_index()

print(df1.head())
print(df2.head())
print(df)

# create a dictionary containing the patient ID as key
# and a list of cluster labels as value
labels = dict()

for id in range(1, len(df)):
    index = df["fox_insight_id"][id]
    if index not in labels:
        labels[index] = []
    labels[index].append(df[cluster_column][id])

data = []
for key in labels.keys():
    l = []
    l.append(key)
    l = l + labels[key]
    data.append(l)

print(data)

# save the new data frame in a new table
progression_df = pd.DataFrame(data).set_index(0)
print(progression_df)
progression_df.to_csv('fox_data_clusters_progression_k3.csv')

