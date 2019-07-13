"""
Returns statistics about dataset
"""


import os, string

corpora_path = "../Datasets/"  # path to original corpora folder
input_dir_names = ["CZ", "DE", "IT", "EN"]  # names of input folders

files = os.listdir(corpora_path)

for i in range(0, len(input_dir_names)):
    files = os.listdir(os.path.join(corpora_path, input_dir_names[i]))  # original files
    number_of_docs = 0
    number_of_words = 0
    num_a1 = 0
    num_a2 = 0
    num_b1 = 0
    num_b2 = 0
    num_c1 = 0
    num_c2 = 0
    for file in files:
        if file.endswith(".txt"):
            number_of_docs += 1
            content = open(os.path.join(corpora_path, input_dir_names[i], file), "r").read()

            number_of_words += sum([word.strip(string.punctuation).isalpha() for word in content.split()])
            cefr = file.split(".txt")[0].split("_")[-1]  # language level

            if cefr == 'A1':
                num_a1 += 1
            if cefr == 'A2':
                num_a2 += 1
            if cefr == 'B1':
                num_b1 += 1
            if cefr == 'B2':
                num_b2 += 1
            if cefr == 'C1':
                num_c1 += 1
            if cefr == 'C2':
                num_c2 += 1

    print('Language:', input_dir_names[i])
    print('Number of documents:', number_of_docs)
    print('Average number of words:', number_of_words / number_of_docs)
    print('A1: \t', num_a1)
    print('A2: \t', num_a2)
    print('B1: \t', num_b1)
    print('B2: \t', num_b2)
    print('C1: \t', num_c1)
    print('C2: \t', num_c2)
    print()
