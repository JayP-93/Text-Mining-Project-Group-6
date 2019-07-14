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
    number_of_words = [0]*6
    num_doc_label = [0]*6
    label_list = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    for file in files:
        if file.endswith(".txt"):
            number_of_docs += 1
            content = open(os.path.join(corpora_path, input_dir_names[i], file), "r", encoding="utf-8").read()
            cefr = file.split(".txt")[0].split("_")[-1]  # language level
            index = label_list.index(cefr)
            number_of_words[index] += sum([word.strip(string.punctuation).isalpha() for word in content.split()])
            num_doc_label[index] += 1

    print('Language:', input_dir_names[i])
    print('Number of documents:', number_of_docs)
    for j, label in enumerate(label_list):
        average_words = 0
        if not num_doc_label[j]:
            print('%s : \t %d , average words: %3.2f' % (label, 0, 0))
        else:
            print('%s : \t %d , average words: %3.2f' % (label, num_doc_label[j], number_of_words[j]/num_doc_label[j]))
    print()
