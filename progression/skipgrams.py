import pandas as pd
import math

# read file containing all patient visits with their cluster labels
df = pd.read_csv('fox_data_clusters_progression_k3.csv')
df = df.drop(df.columns[0], axis=1)
print(df.head())

# map the cluster labels to start from 1, instead of 0
B = pd.DataFrame({'Code': [0, 1, 2],
                  'Value': [1, 2, 3]})
df = df.replace(B.set_index('Code')['Value'])
print(df.head())

# method for skipgrams
def kskipngrams(sentence,k,n):
    """Assumes the sentence is already tokenized into a list"""
    if n == 0 or len(sentence) == 0:
        return None
    grams = []
    for i in range(len(sentence)-n+1):
        grams.extend(initial_kskipngrams(sentence[i:],k,n))
    return grams

def initial_kskipngrams(sentence,k,n):
    if n == 1:
        return [[sentence[0]]]
    grams = []
    for j in range(min(k+1,len(sentence)-1)):
        kmjskipnm1grams = initial_kskipngrams(sentence[j+1:],k-j,n-1)
        if kmjskipnm1grams is not None:
            for gram in kmjskipnm1grams:
                grams.append([sentence[0]]+gram)
    return grams

# collect all sequences (c1,c2,...,cn) to a list
list = df.values.tolist()
# initialize list for storing sequences strings
str_skipgrams = []
# length of ngrams (some can be shorter due to the different lenghts of visits 10-15)
n = 4

# compute skip grams
for seq in list:
    result = kskipngrams(seq, 3, 4)
    #print(result)

    """Convert skipgrams into strings ( '11111', '10111'...)"""
    for integers in result:
        strings = [str(int(integer)) for integer in integers if math.isnan(integer) == False ]
        if len(strings) == n:
            str_skipgrams.append("".join(strings))

# create data frame with frequences of ngrams
stats = pd.Series(str_skipgrams).value_counts().to_frame()
stats.columns = ['freq']
# remove ngrams which occurred less than 1500.
stats = stats[stats['freq'] >= 1500]
print(stats)

"""Create a histogram"""
import matplotlib.pyplot as plt
stats.plot(kind ='bar')
plt.ylabel('Number of cluster crossings')
plt.xlabel('Sequences of cluster crossings')
plt.show()

