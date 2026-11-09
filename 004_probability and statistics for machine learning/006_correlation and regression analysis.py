# # exercise 1: calculate correlation between features
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
# df = pd.read_csv(url)
#
# del df['species']
#
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap="viridis")
# plt.title('Feature Correlation')
# plt.show()


# exercise 2: fit a Simple Linear Regression Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
x = np.random.rand(230,1) * 10
y = 3 * x + np.random.randn(230, 1) * 2
model = LinearRegression()
model.fit(x, y)
slope = model.coef_[0][0]
intercept = model.intercept_[0]
r_squared = model.score(x, y)

plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.legend()
plt.title('Linear Regression')
plt.show()