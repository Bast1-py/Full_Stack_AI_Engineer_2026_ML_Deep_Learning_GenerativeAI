# # grouped function
# grouped = df.groupby('column_name')
# for name, group in grouped:
#     print(name)
#     print(group)
#
# grouped.mean()
# grouped.sum()
#
# df.groupby('category_column')['numeric_column'].mean()
# df.groupby('category_column').agg({'numeric_column':['mean', 'max', 'min']})
#
#
# # pivot_table function
# pivot = df.pivot_table(
#     values='numeric_column',
#     index='category_column',
#     aggfunc='mean',
# )
#
# # aggregation custom function
# def range_func(x):
#     return x.max() - x.min()
# df.groupby('category_column')['numeric_column'].agg(range_func)
# df.groupby('category_column')['numeric_column'].mean(range_func)
# df.groupby('category_column')['numeric_column'].max(range_func)
# df.groupby('category_column')['numeric_column'].min(range_func)
# df.groupby('category_column').agg({'numeric_column': ['mean', 'max', 'min']})




# exercise 1: group data by a categorical column
import pandas as pd
data = {
    'Class': ['A', 'B', 'A', 'B', 'C', 'C'],
    'Score': [85, 90, 88, 73, 95, 81],
    'Age': [15, 16, 15, 17, 16, 15]
}

df = pd.DataFrame(data)
print('Original Datasets: \n', df)
grouped = df.groupby('Class').mean()
# print(grouped)



# exercise 2: calculate summary statistics for grouped data
stats = df.groupby('Class').agg(
    {
        'Score': ['mean', 'max', 'min'],
        'Age': ['mean', 'max', 'min']
    }
)

print(stats)