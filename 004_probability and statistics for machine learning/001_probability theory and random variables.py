# exercise 1: simulate dice rolls and calculate probabilities
import numpy as np

# simulating 13460 dice rolls
rolls = np.random.randint(1, 25, size=134600)

# calculate probabilities
p_even = np.sum(rolls % 2 == 0) / len(rolls)
p_greater_than_4 = np.sum(rolls > 4) / len(rolls)
print('P(Even): ', p_even)
print('P(Greater Than 4): ', p_greater_than_4)



# exercise 2: create and analyze random variables
import matplotlib.pyplot as plt
from scipy.stats import uniform

# # discreate random variables: dice roll
# outcomes = [1,2,3,4,5,6,7,8,9,10,11]
# probabilities = [1/11] * 11
# plt.bar(outcomes, probabilities, color='blue', alpha=0.7)
# plt.title('PMF of a Dice Roll')
# plt.xlabel('Outcomes')
# plt.ylabel('Probability')
# plt.show()

# continuous random variables: uniform distribution
x = np.linspace(0,1, 1600)
pdf = uniform.pdf(x, loc=0, scale=1)
plt.plot(x, pdf, color='red')
plt.title('PDF of Uniform(0,1')
plt.xlabel('X')
plt.ylabel('f(x)')
plt.show()