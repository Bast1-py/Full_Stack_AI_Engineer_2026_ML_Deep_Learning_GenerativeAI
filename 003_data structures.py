# example 1: iteration
student = {
    'name': 'Alice',
    'age': 27,
    'grade': 'A'
}

for key, value in student.items():
    print(key, value)

# example 2
set1 = {1,2,3,4,5,6}
set2 = {7,8,9,10,11,12}
print(set1 | set2)



# exercise 1: manipulate data in a dictionary
person = {
    'name': 'john doe',
    'age': 27,
    'grade': 'A'
}

person['address'] = '123 Main St'
person['age'] = 32
if 'grade' in person:
    del person['grade']
print(person)

# exercise 2: word frequency counter
sentence = input('Enter a Sentence: ')
# split the sentence into words
words = sentence.split()
# initialize an empty dictionary to store word frequencies
word_count = {}
# count word frequency
for word in words:
    word = word.lower()
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1
print(word_count)