# hands-on exercise: 1. load a dataset and explore its features, identifying categorical and numerical features
# 2. plan which feature engineering techniques might be most suitable for the dataset

import pandas as pd


url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

print('Datasets Info: \n')
print(df.info())
print('\n Datasets Preview: \n')
print(df.head())

# seperate features
cat_features = df.select_dtypes(include=['object']).columns
num_features = df.select_dtypes(include=['int64', 'float64']).columns

print('\nCategorical Features: ', cat_features.tolist())
print('\nNumerical Features: ', num_features.tolist())

# display summary of categorical features
print('\nCategorical Features Summary: \n')
for col in cat_features:
    print(f'{col}:\n', df[col].value_counts(), '\n')


# display summary of numerical features
print(df[num_features].describe())