"""
Saves information about errors (using Language check) in DE and IT data to text files.
The main purpose: knowledge about error statistics in the data for DE and IT.
The results (text files) are in the features/ folder in this repo
"""

import language_check
import os


def write_featurelist(file_path, some_list):
    """
    Creates file with data from list (each element of list is a new line in text file)
    :param file_path: path to the file
    :param some_list: list with data
    """
    fh = open(file_path, "w")
    for item in some_list:
        fh.write(item)
        fh.write("\n")
    fh.close()


def error_stats(inputpath, lang, output_path):
    """
    Creates three text files with information of different errors in input texts.
    :param inputpath: path to folder with input data
    :param lang: string with name of language, e.g. 'de'
    :param output_path: path to the output text files
    """
    files = os.listdir(inputpath)  # input files
    checker = language_check.LanguageTool(lang)
    rules = {}
    locqualityissuetypes = {}
    categories = {}

    for file in files:
        if file.endswith(".txt"):
            text = open(os.path.join(inputpath, file)).read()
            matches = checker.check(text)
            for match in matches:
                rule = match.ruleId
                loc = match.locqualityissuetype
                cat = match.category
                rules[rule] = rules.get(rule, 0) + 1
                locqualityissuetypes[loc] = locqualityissuetypes.get(loc, 0) + 1
                categories[cat] = categories.get(cat, 0) + 1

    write_featurelist(output_path + lang + "-rules.txt", sorted(rules.keys()))
    write_featurelist(output_path + lang + "-locquality.txt", sorted(locqualityissuetypes.keys()))
    write_featurelist(output_path + lang + "-errorcats.txt", sorted(categories.keys()))


def main():
    inputpath_de = "/home/bangaru/GitProjects/CrossLingualScoring/Datasets/DE/"
    inputpath_it = "/home/bangaru/GitProjects/CrossLingualScoring/Datasets/IT/"

    error_stats(inputpath_de, "de", "../features/")
    error_stats(inputpath_it, "it", "../features/")


if __name__ == "__main__":
    main()
