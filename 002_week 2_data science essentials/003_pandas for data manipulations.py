import pandas as pd

s = pd.Series([10,22,34], index=['a', 'b', 'c'])
# print(s)

data = {
    'Name': ['Alice', 'Doe'],
    'Age': [25,30]
}

df = pd.DataFrame(data)
# print(df)


# exercise 1: load and explore a sample dataset
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
# print('First 6 rows: \n', df.head(6))
# print('\nLast 6 rows: \n', df.tail(6))
# print(df.info())
# print(df.describe())



# exercise 2: select specific columns and filter rows
selected_columns = df[['species', 'sepal_length']]
# print('Selected Columns: \n', selected_columns)

filtered_rows = df[(df['sepal_length'] > 5.0) & (df['species'] == 'setosa')]
print('Filteres Rows: \n', filtered_rows)