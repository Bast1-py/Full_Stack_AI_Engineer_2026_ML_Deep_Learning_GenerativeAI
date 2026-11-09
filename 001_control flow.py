# # example 1: checking and condition
# num = 120
# if num > 0:
#     print('Positive Number')
# elif num == 0:
#     print('Zero')
# else:
#     print('Negative Number')
#
# # example 2: nested conditions
# age = 27
# if age > 18:
#     if age < 30:
#         print('You are a young adult')
#     else:
#         print('Adult')
#
# # example 3: for-loops
# ## loop through a list
# fruits = ['apple', 'banana', 'cherry']
# for fruit in fruits:
#     print(fruit)
#
# ## loop with range
# for i in range(12):
#     print(i)
#
# # example 4: while-loops
# count = 5
# while count > 0:
#     print(count)
#     count -= 1
# print('Outside While Loop')
#
# # example 5: break and continue
# ## break statement
# for i in range(29):
#     if i == 11:
#         break
#     print(i)
# print('Outside For Loop')
#
# ## continue statement
# for i in range(29):
#     if i == 11:
#         continue
#     print(i)
# print('Outside For Loop')
#
# for i in range(11):
#     if i % 2 == 0:
#         continue
#     print(i)


# # Exercise 1: check if a number is prime or not
# num = int(input('Enter a Number: '))
# if num > 1:
#     for i in range(2, int(num**0.5) + 1):
#         if num % i == 0:
#             print(f'{num} is not a prime number')
#             break
#     else:
#         print(f'{num} is a prime number')
# else:
#     print(f'{num} is not a prime number')


# Exercise 2: Create a Menu-Driven Calculator
def add(a,b):
    return a+b
def subtract(a,b):
    return a-b
def multiply(a,b):
    return a*b
def divide(a,b):
    if b != 0:
        return a/b
    else:
        return 'Division by zero is not allowed'
while True:
    print('\nMenu:')
    print('1. Add')
    print('2. Subtract')
    print('3. Multiply')
    print('4. Divide')
    print('5. Exit')

    choice = input('Enter Your Choice: ')

    if choice == '5':
        print('Exiting Program.')
        break
    num1 = float(input('Enter first number: '))
    num2 = float(input('Enter second number: '))

    if choice == '1':
        print (f'Result: ', add(num1, num2))
    elif choice == '2':
        print (f'Result: ', subtract(num1, num2))
    elif choice == '3':
        print (f'Result: ', multiply(num1, num2))
    if choice == '4':
        print (f'Result: ', divide(num1, num2))
    else:
        print('Invalid Choice. Please try again.')
