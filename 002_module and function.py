# def add_numbers(a,b):
#     return a + b
# result = add_numbers(53,22)
# print('Sum: ', result)
#
# # example 1: Local scope
# def greet():
#     message = 'Hello World'
#     print(message)
# greet()
#
#
# # example 2: Global scope
# greeting = 'hi'
# def say_hello():
#     print(greeting + ' From inside the function')
# say_hello()
# print(greeting + ' From outside the function')
#
#
# import math
# print(math.sqrt(16))
#
# from math import sqrt
# print(sqrt(16))
#
# import math as m
# print(m.sqrt(16))



# exercise 1: create a function to calculate factorials
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

def print_factorial(n):
    result = factorial(n)
    print(f'The Factorial of {n} is {result}')
print_factorial(5)