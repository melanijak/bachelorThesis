import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read the file with baseline visits and cluster labels
df = pd.read_csv('fox_data_clusters_k3.csv', index_col=0)
print(df.head())

# read file with remaining visits
pred = pd.read_csv('fox_data_remaining_10do15.csv')
print(pred.head())

target = 'Cluster'

# prepare X and y
X = df.loc[:, df.columns != target]
X = df.drop('fox_insight_id', 1)
y = df[target]

# prepare the data set for prediction
Xpred = pred.drop('fox_insight_id', 1)

# define the RF model
rf = RandomForestClassifier(n_estimators=20, random_state=0)
# separate the training and testing set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
rf.fit(Xtrain, ytrain)

# print the accuracy of the model on the test data set
ytest_pred = rf.predict(Xtest)
print(accuracy_score(ytest, ytest_pred))

# predict the cluster labels of the remaining visits
y_pred = rf.predict(Xpred)
pred[target] = y_pred

print(pred.head())
# save the file with the predicted cluster labels in a new table
pred.to_csv('fox_data_clusters_k3_prediction.csv')

#import shap
#shap_values = shap.TreeExplainer(rf).shap_values(Xtrain)
#shap.summary_plot(shap_values, Xtest, plot_type="bar")
