"""
Get error features for a given text file, by passing it through Language Tool,
and counting the occurrences of different rule matches. Saves results as numpy array.
The results (.npy files) are in the features/ folder in this repo
"""

import language_check
import os, numpy


def loadfeatures(feature_file_path):
    """
    Creates dictionary from feature file
    :param feature_file_path: path to feature file
    :return: dictionary (all values equal to zero)
    """
    feature_dict = {}
    fh = open(feature_file_path)
    for line in fh:
        feature_dict[line.strip()] = 0
    fh.close()
    return feature_dict


def reset_dict(d):
    """
    Reset dictionary values to zero after finishing each file.
    :param d: dictionary
    :return: dictionary with zero values
    """
    d = dict((k, 0) for k in d)
    return d


#
def get_file_feats(rules, locqualityissuetypes, categories):
    """
    Combine all three dictionaries of features to get features for one file
    :param rules: dicionary of rules
    :param locqualityissuetypes: dictionary of issue types
    :param categories: dictionary of categories
    :return: new dictionary with all features (all input dictionaries combined)
    """
    big_dict = {}
    big_dict.update(rules)
    big_dict.update(locqualityissuetypes)
    big_dict.update(categories)
    return [big_dict[key] for key in sorted(big_dict.keys())]


def get_all_features(dirpath, lang, categories, locqualityissuetypes, rules):
    """
    Get features and language levels for a language
    :param dirpath: path to the folder with input files
    :param lang: string with name of language, e.g. 'de'
    :param categories: dictionary of categories
    :param locqualityissuetypes: dictionary of issue types
    :param rules: dicionary of rules
    :return: two lists: with language levels and with features of all files in the folder
    """
    files = os.listdir(dirpath)
    all_feats = []  # list of feature dictionaries of all files in the folder
    all_cats = []  # list of language levels of all files in the folder
    checker = language_check.LanguageTool(lang)
    for f in files:
        if f.endswith(".txt"):
            text = open(os.path.join(dirpath, f)).read()
            matches = checker.check(text)
            for match in matches:
                rule = match.ruleId
                loc = match.locqualityissuetype
                cat = match.category
                rules[rule] = rules.get(rule, 0) + 1
                locqualityissuetypes[loc] = locqualityissuetypes.get(loc, 0) + 1
                categories[cat] = categories.get(cat, 0) + 1
            all_feats.append(get_file_feats(rules, locqualityissuetypes, categories))
            all_cats.append(f.split(".txt")[0][-2:])

            # reset all values in a dictionmaries
            rules = reset_dict(rules)
            categories = reset_dict(categories)
            locqualityissuetypes = reset_dict(locqualityissuetypes)

    return all_cats, all_feats


def main():
    # DE
    de_errorcats = loadfeatures("../features/de-errorcats.txt")
    de_locquality = loadfeatures("../features/de-locquality.txt")
    de_rules = loadfeatures("../features/de-rules.txt")

    de_corpuspath = "/home/bangaru/GitProjects/CrossLingualScoring/Datasets/DE/"  # path to dataset
    de_cats, de_feats = get_all_features(de_corpuspath, "de", de_errorcats, de_locquality, de_rules)
    print(len(de_cats), len(de_feats), len(de_feats[0]), len(de_feats[1]))
    numpy.save("../features/defeats", numpy.array(de_feats))
    numpy.save("../features/decats", numpy.array(de_cats))

    # IT#
    it_errorcats = loadfeatures("../features/it-errorcats.txt")
    it_locquality = loadfeatures("../features/it-locquality.txt")
    it_rules = loadfeatures("../features/it-rules.txt")

    it_corpuspath = "/home/bangaru/GitProjects/CrossLingualScoring/Datasets/IT/"  # path to dataset
    it_cats, it_feats = get_all_features(it_corpuspath, "it", it_errorcats, it_locquality, it_rules)
    print(len(it_cats), len(it_feats), len(it_feats[0]), len(it_feats[1]))
    numpy.save("../features/itfeats", numpy.array(it_feats))
    numpy.save("../features/itcats", numpy.array(it_cats))


if __name__ == "__main__":
    main()
