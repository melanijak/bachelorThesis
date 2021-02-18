from scipy import stats
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

# read the file which contains all patient visits together with their cluster assignments
df = pd.read_csv('NMF/fox_data_clusters_10do15.csv', index_col = 0)
target = 'Cluster'
X = df.drop(target, 1)
columns = X.columns
print(df.head())

# define between which cluster label to perform test
cluster_value1 = 0
cluster_value2 = 1

# extract rows of the df which has cluster_valuex the assigned cluster
clusterFirst = df[df[target] == cluster_value1]
clusterSecond = df[df[target] == cluster_value2]

dictionary = dict()
# iterate through all columns of the data
for s in columns:
    # extract vectors with column s for the chosen cluster label
    c1 = clusterFirst[s]
    c2 = clusterSecond[s]
    # compute the Kolmogorov-Smirnov statistic on 2 vectors
    array = np.array(stats.ks_2samp(c1, c2))
    # compute mean value of the symptom of chosen cluster label
    m1 = clusterFirst[s].mean()
    m2 = clusterSecond[s].mean()
    # compute std value of the symptom of chosen cluster label
    std1 = clusterFirst[s].std()
    std2 = clusterSecond[s].std()

    dictionary[s] = array[1], np.round(m1, 3), np.round(std1, 3), np.round(m2, 3), np.round(std2, 3)

# save the results in a new Data Frame
dfRez = pd.DataFrame.from_dict(dictionary, orient='index')
dfRez.columns = ['PiValue', 'MeanFirst', 'STDFirst', 'MeanSecond', 'STDSecond']
# sort values according to the p value
dfRez = dfRez.sort_values('PiValue')
print(dfRez)

# print the top 50 columns and values (latex table format)
n = 0
for index, row in dfRez.iterrows():
    if n == 50:
        break
    print('%s & %s(%s) & %s(%s) \\\\' % (index, format(row['MeanFirst'], '.3f'),
                                          format(row['STDFirst'], '.3f'),  format(row['MeanSecond'], '.3f'),
                                          format(row['STDSecond'], '.3f')))
    n += 1
