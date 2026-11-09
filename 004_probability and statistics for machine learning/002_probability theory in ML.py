# exercise 1: plot and explore various distributions
from scipy.stats import skew, kurtosis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
df = pd.read_csv(url)

feature = df['sepal_length']
print('Skewness: ', skew(feature))
print('Kurtosis: ', kurtosis(feature))

sns.histplot(feature, kde=True)
plt.title('Distribution of Sepal Length')
plt.show()