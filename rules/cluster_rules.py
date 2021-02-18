import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree

# read file with baseline visits and their cluster assignments
df = pd.read_csv('fox_data_clusters_k3.csv', index_col=0)
print(df.head())

# prepare X and y for model
target = 'Cluster'
X = df.drop(['fox_insight_id', target], 1)
y = df.iloc[:,-1:]

# split data into training and testing
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42)
# initialize and fit model
clf = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=18)
clf.fit(Xtrain, ytrain)

# method for printing rules (latex table format)
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    pathto=dict()

    global k
    k = 0
    def recurse(node, depth, parent):
        global k
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            s= "{} $\leq$ {} ".format( name, round(threshold,2), node )
            if node == 0:
                pathto[node]=s
            else:
                pathto[node]=pathto[parent]+' \& ' +s

            recurse(tree_.children_left[node], depth + 1, node)
            s="{} $>$ {}".format( name, round(threshold,2))
            if node == 0:
                pathto[node]=s
            else:
                pathto[node]=pathto[parent]+' \& ' +s
            recurse(tree_.children_right[node], depth + 1, node)
        else:
            k=k+1
            print(k,')',pathto[parent], ' class = ', np.argmax(tree_.value[node]), tree_.value[node])
    recurse(0, 1, 0)

# print rules
tree_to_code(clf, list(X.columns))

# predict and print the accuracy score of the DT
res_pred = clf.predict(Xtest)
score = accuracy_score(ytest, res_pred)
print(score)

# print the default class and the distributions of others
print('default class:')
print(max(df[target].value_counts())/len(df))
print('cluster distributions: ')
print(df[target].value_counts())

#import shap
#shap_values = shap.TreeExplainer(clf).shap_values(Xtest)
#shap.summary_plot(shap_values, Xtest, plot_type="bar")
