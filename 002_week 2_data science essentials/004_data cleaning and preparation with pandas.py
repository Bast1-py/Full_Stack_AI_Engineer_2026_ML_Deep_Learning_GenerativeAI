# exercise 1: clean a dataset by handling missing values and renaming columns
import pandas as pd
import numpy as np

data = {
    'Name': ['Alice', 'john', 'Bob', np.nan, 'Doe'],
    'Age': [25, np.nan, 27, 30, 32],
    'Score': [85, 87, np.nan, 89, 91]
}

df = pd.DataFrame(data)
# print('Original Dataset: \n', df)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Score'] = df['Score'].interpolate()
# print('\nDataset: \n', df)

df = df.rename(
    columns={
        'Name': 'Student_name',
        'Score': 'Exam:score'
    }
)

# print('\nDataset: \n', df)


# exercise 2: merge two datasets and perform dara transformations
df1 = pd.DataFrame(
    {
        'ID': [1,2,3,4],
        'Name': ['Alice', 'Charlie', 'Doe', 'Bob'],
        'Age': [26, 30, 31, 35]
    }
)

df2 = pd.DataFrame(
    {
        'ID': [1,2,3,4],
        'Score': [85,91,93,94]
    }
)

print('Dataset 1: \n', df1)
print('Dataset 2: \n', df2)
merged = pd.merge(
    df1, df2, how='inner', on='ID'
)
print('\nMerged Datasets: \n', merged)

merged['Score_percentage'] = (merged['Score'] / 100) * 100
print('\nTransformed Datasets \n', merged)