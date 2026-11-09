# # example 1: slicing
# text = 'Python Programming'
# print(text[0:6])
# print(text[-11:])
#
# # example 2: formatting
# name = 'Alice'
# age = 23
# print(f'My name is {name} and I am {age} years old.')
#
# # example 3: common string methods
# sentence = 'Python Programming, is, fun'
# words = sentence.split(',')
# print(words)
#
# new_sentence = '|'.join(words)
# print(new_sentence)
#
# text = 'I Love Java'
# updated_text = text.replace('Java', 'Python')
# print(updated_text)
#
# messy = "     Hello, World!     "
# print(messy)
# cleaned_text = messy.strip()
# print(cleaned_text)
#
# # example 4: regular expressions for pattern matching
# import re
# text = 'Contact me at 123-456-7890'
# digits = re.findall(r'\d+', text)
# print(digits)
#
# updated_text = re.sub(f'\d', 'X', text)
# print(updated_text)


# exercise 1: create a text cleaner
import re
def clean_text(text):
    #remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    #remove extra spaces
    text = ' '.join(text.split())
    # convert to lowercase
    return text.lower()
input_text = "    Hello, World.!!! Welcome to Python, Programming..."
cleaned_text = clean_text(input_text)
print('Cleaned Text: ', cleaned_text)

# exercise 2: check if a string is a palindrome
def is_palindrome(text):
    text = ''.join(char.lower() for char in text if char.isalnum())
    return text == text[::-1]
input_text = input('Enter a String: ')
if is_palindrome(input_text):
    print('"{input_text}" is a palindrome.')
else:
    print(f'"{input_text}" is not a palindrome.')