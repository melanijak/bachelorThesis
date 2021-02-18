import pandas as pd

# read file containing the symptom/medication ranges
df = pd.read_csv('../FoxInsightValues.csv')
print(df.head())

norm_dict = dict()

# create a dictionary containing symptoms as keys
# and list of possible values as value
for name, value in zip(df['variable'], df['value']):
    if name not in norm_dict:
        norm_dict[name] = []
    norm_dict[name].append(value)

print(norm_dict)

# save only the min and max value per symptom
for key in norm_dict.keys():
    min1 = min(norm_dict[key])
    max1 = max(norm_dict[key])
    norm_dict[key] = [min1, max1]

# read the max and min age from the original data set
df = pd.read_csv('fox_data.csv')
norm_dict['age'] = [df['age'].min(), df['age'].max()]
print(norm_dict)

# make a copy of the original data set
result = df.copy()

for column in df.columns:
        min_value = norm_dict[column][0]
        max_value = norm_dict[column][1]
        # replace the column with a column with normalized values
        result[column] = (df[column] - min_value) / (max_value - min_value)

print(result)
result.to_csv('norm/fox_data_norm.csv')



