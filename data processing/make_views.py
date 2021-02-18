import pandas as pd

# read file with the merged symptoms data set
df = pd.read_csv('fox_data_baseline_10do15.csv')
dir = 'MVC/'

# divide table in separate views and save them
col1 = [col for col in df if col.startswith('Move')]
col1 = ['fox_insight_id'] + col1
df1 = df[col1]
df1.loc[:, 'MoveSUM'] = df1.sum(axis=1)
df1.to_csv(dir+'PD_Movement.csv', index = False)

col1 = [col for col in df if col.startswith('NonMove')]
col1 = ['fox_insight_id'] + col1
df1 = df[col1]
df1.loc[:, 'NonMoveSUM'] = df1.sum(axis=1)
df1.to_csv(dir+'PD_NonMovement.csv', index = False)

col1 = [col for col in df if col.startswith('MedsCurr')]
col1 = ['fox_insight_id'] + col1
df1 = df[col1]
df1.loc[:, 'MedsCurrSUM'] = df1.sum(axis=1)
df1.to_csv(dir+'PD_MedsPDCurrent.csv', index = False)

col1 = [col for col in df if col.startswith('MedsOther') or col.startswith('MedsVit') or col.startswith('MedPDProced')]
col1 = ['fox_insight_id'] + col1
df1 = df[col1]
df1.loc[:, 'MedsOtherSUM'] = df1.sum(axis=1)
df1.to_csv(dir+'PD_MedsOther.csv', index = False)

col1 = [col for col in df if col.startswith('Leisure') or col.startswith('House')
        or col.startswith('Work') or col.startswith('Strength') or 'Sport' in col]
col1 = ['fox_insight_id'] + col1
df1 = df[col1]
df1.loc[:, 'ActivitiesSUM'] = df1.sum(axis=1)
df1.to_csv(dir+'PD_PhysicalActivities.csv', index = False)

col1 = [col for col in df if col.startswith('Daily')]
col1 = ['fox_insight_id'] + col1
df1 = df[col1]
df1.loc[:, 'DailySUM'] = df1.sum(axis=1)
df1.to_csv(dir+'PD_CognitionDaily.csv', index = False)

col1 = ['Mobility','Care','Active','Pain','Anxious']
col1 = ['fox_insight_id'] + col1
df1 = df[col1]
df1.loc[:, 'PhysicalSUM'] = df1.sum(axis=1)
df1.to_csv(dir+'PD_PhysicalExperience.csv', index = False)

