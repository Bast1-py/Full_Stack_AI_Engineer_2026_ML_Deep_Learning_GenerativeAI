# with open('Sample.txt', 'w') as file:
#     # content = file.readlines()
#     # print(content)
#     file.write('Hello, World!')
#     file.writelines('Alice', 'Bob', 'Cherry')

# file is automatically closed
try:
    with open('Sample.txt', 'r') as file:
        content = file.read()
except FileNotFoundError:
    print('File is Not Found')


# exercise 1: count words and lines in a file
def count_words_and_lines(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            line_count = len(lines)
            word_count = sum(len(line.split()) for line in lines)
            print(f'Number of Lines: {line_count}')
            print(f'Number of Words: {word_count}')
    except FileNotFoundError:
        print(f'File {filename} not found!')

count_words_and_lines('Sample.txt')

# exercise 2: write and read a list of items
def write_items_to_file(filename, items):
    with open(filename, 'w') as file:
        for item in items:
            file.write(item + '\n')
def read_items_from_file(filename):
    try:
        with open(filename, 'r') as file:
            items = file.readlines()
            print('Items in the file: ')
            for item in items:
                print(item.strip())
    except FileNotFoundError:
        print(f'File {filename} not found!')
fruits = [
    'Apple',
    'Banana',
    'Cherry',
    'Dates'
]
write_items_to_file('fruits.txt', fruits)
read_items_from_file('items.txt')