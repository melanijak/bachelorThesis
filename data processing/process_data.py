import pandas as pd
from sklearn.impute import KNNImputer

# read the downloaded file from Fox Insight
df = pd.read_csv('fox_data.csv')

# preserve only patients with diagnosed PD
df = df[df.CurrPDDiag != 0]
df = df.drop('CurrPDDiag', 1)

# info about data set
print("Number of patients: ", len(df['fox_insight_id'].value_counts()))
print("Number of rows: ", len(df))
print("Number of variables: ", len(df.columns))
dfreq = df['fox_insight_id'].value_counts()
# print frequences of patient visits
print(dfreq.value_counts())

# print missing values stats
print(df.isnull().sum(axis=1).value_counts())

# remove rows with more missing values
df = df[df.isnull().sum(axis=1) < 50]

# filter patients which have between 10 and 15 visits
df1 = df[df.groupby("fox_insight_id")['fox_insight_id'].transform('size') <= 10]
df = df1[df1.groupby("fox_insight_id")['fox_insight_id'].transform('size') >= 15]

# fill remaining missing values with mean of each symptom
df = df.fillna(df.mean())

# get indexes of rows containing the baseline (first) visit of the patient
idx = df.groupby(['fox_insight_id'])['age'].transform(min) == df['age']

df1 = df[idx]
dfreq= df1['fox_insight_id'].value_counts()
print(dfreq.value_counts())

# print info of the new data frame
print("Number of rows: ", len(df1))
print("Number of variables: ", len(df1.columns))

# save the table and use it for clustering
df1.to_csv('views/fox_data_BL_10do15.csv', index=False)

# create a new data frame with the remaining visits
rows = df.index[idx]
d2 = df.drop(rows)
d2 = d2.fillna(d2.mean())
d2.to_csv('fox_data_remaining_10do15.csv', index=False)



