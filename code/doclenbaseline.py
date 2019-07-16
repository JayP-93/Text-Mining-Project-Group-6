"""
Perform monolingual, multilingual and crosslingual classification,
using length of document (number of words) as a feature.
The results are baseline for next parts.
"""

import os
import string

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.svm import LinearSVC

seed = 1234


def getdoclen(conllufilepath):
    """
    Returns number of words in the text file
    :param conllufilepath: path to the file
    :return: number of words in the file
    """
    fh = open(conllufilepath, encoding="utf-8")
    allText = []
    sent_id = 0
    for line in fh:
        if line == "\n":
            sent_id = sent_id + 1
        elif not line.startswith("#") and line.split("\t")[3] != "PUNCT":
            word = line.split("\t")[1]
            allText.append(word)
    fh.close()
    return len(allText)


def getfeatures(dirpath):
    """
    Returns lists with features (document lengths and language levels) of the text files
    :param dirpath: path to folder with text files
    :return: two lists: with documents lengths and with language levels
    """
    files = os.listdir(dirpath)
    cats = []  # list with language levels
    doclenfeaturelist = []  # list with documents lengths
    for filename in files:
        if filename.endswith(".txt"):
            doclenfeaturelist.append([getdoclen(os.path.join(dirpath, filename))])
            cats.append(filename.split(".txt")[0].split("_")[-1])
    return doclenfeaturelist, cats


def singleLangClassificationWithoutVectorizer(train_vector, train_labels):
    """
    Classification using different classifiers and k-fold validation.
    Prints out metrics like cross validation score, confusion matrix and F1 score
    :param train_vector: list with features
    :param train_labels: list with labels
    """
    k_fold = StratifiedKFold(10, random_state=seed)  # split data into tran/test
    classifiers = [RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=seed),
                   LinearSVC(class_weight="balanced", random_state=seed),
                   LogisticRegression(class_weight="balanced", random_state=seed)]
    # Not useful: SVC with kernels - poly, sigmoid, rbf.
    for classifier in classifiers:
        print(classifier)
        cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
        predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold)
        print('Cross validation score:\n', cross_val, '\n')
        print('Average cross validation score: ', sum(cross_val) / float(len(cross_val)), '\n')
        print('Confusion matrix:\n', confusion_matrix(train_labels, predicted), '\n')
        print('F1 score: ', f1_score(train_labels, predicted, average='macro'), '\n')


def crossLangClassificationWithoutVectorizer(train_vector, train_labels, test_vector, test_labels):
    """
    Classification using different classifiers. Performs training and testing on different data sets.
    Prints out metrics like accuracy, confusion matrix and F1 score
    :param train_vector: list with features for training
    :param train_labels: list with labels for training
    :param test_vector: list with features for testing
    :param test_labels: list with labels for testing
    """

    # Begin of Cristina's code. Comment to delete at the end.
    diff_labels = set(test_labels) - set(train_labels)
    if diff_labels:
        indices = [i for i, x in enumerate(test_labels) if x in diff_labels]
        test_vector = [i for j, i in enumerate(test_vector) if j not in indices]
        test_labels = [x for x in test_labels if x not in diff_labels]
    # End of Cristina's code. Comment to delete at the end

    classifiers = [RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=seed),
                   LinearSVC(class_weight="balanced", random_state=seed),
                   LogisticRegression(class_weight="balanced", random_state=seed)]
    for classifier in classifiers:
        classifier.fit(train_vector, train_labels)
        predicted = classifier.predict(test_vector)
        print('Accuracy: ', np.mean(predicted == test_labels, dtype=float), '\n')
        print('Confusion matrix:\n', confusion_matrix(test_labels, predicted), '\n')
        print('F1 score: ', f1_score(test_labels, predicted, average='weighted'), '\n')


def main():
    itdirpath = "../Datasets/IT-Parsed"
    dedirpath = "../Datasets/DE-Parsed"
    czdirpath = "../Datasets/CZ-Parsed"
    endirpath = "../Datasets/EN-Parsed"

    # Monolingual classification baseline
    print("********* Start single language *********")
    print("************DE baseline:****************")
    defeats, delabels = getfeatures(dedirpath)
    singleLangClassificationWithoutVectorizer(defeats, delabels)
    print("************IT baseline:****************")
    itfeats, itlabels = getfeatures(itdirpath)
    singleLangClassificationWithoutVectorizer(itfeats, itlabels)
    print("************CZ baseline:****************")
    czfeats, czlabels = getfeatures(czdirpath)
    singleLangClassificationWithoutVectorizer(czfeats, czlabels)
    print("************EN baseline:****************")
    enfeats, enlabels = getfeatures(endirpath)
    singleLangClassificationWithoutVectorizer(enfeats, enlabels)
    print("********* End single language *********")

    print("********* Start cross language *********")
    # Crosslingual classification baseline DE
    print("*** Train with DE, test with IT baseline******")
    crossLangClassificationWithoutVectorizer(defeats, delabels, itfeats, itlabels)
    print("*** Train with DE, test with CZ baseline ******")
    crossLangClassificationWithoutVectorizer(defeats, delabels, czfeats, czlabels)
    print("*** Train with DE, test with EN baseline ******")
    crossLangClassificationWithoutVectorizer(defeats, delabels, enfeats, enlabels)

    # Crosslingual classification baseline IT
    print("*** Train with IT, test with DE baseline******")
    crossLangClassificationWithoutVectorizer(itfeats, itlabels, defeats, delabels)
    print("*** Train with IT, test with CZ baseline ******")
    crossLangClassificationWithoutVectorizer(itfeats, itlabels, czfeats, czlabels)
    print("*** Train with IT, test with EN baseline ******")
    crossLangClassificationWithoutVectorizer(itfeats, itlabels, enfeats, enlabels)

    # Crosslingual classification baseline CZ
    print("*** Train with CZ, test with IT baseline******")
    crossLangClassificationWithoutVectorizer(czfeats, czlabels, itfeats, itlabels)
    print("*** Train with CZ, test with DE baseline ******")
    crossLangClassificationWithoutVectorizer(czfeats, czlabels, defeats, delabels)
    print("*** Train with CZ, test with EN baseline ******")
    crossLangClassificationWithoutVectorizer(czfeats, czlabels, enfeats, enlabels)

    # Crosslingual classification baseline EN
    print("*** Train with EN, test with IT baseline******")
    crossLangClassificationWithoutVectorizer(enfeats, enlabels, itfeats, itlabels)
    print("*** Train with EN, test with CZ baseline ******")
    crossLangClassificationWithoutVectorizer(enfeats, enlabels, czfeats, czlabels)
    print("*** Train with EN, test with DE baseline ******")
    crossLangClassificationWithoutVectorizer(enfeats, enlabels, defeats, delabels)
    print("********* End cross language *********")

    # Multilingual classification baseline
    bigfeats = []
    bigcats = []
    bigfeats.extend(defeats)
    bigfeats.extend(itfeats)
    bigfeats.extend(czfeats)
    bigfeats.extend(enfeats)
    bigcats.extend(delabels)
    bigcats.extend(itlabels)
    bigcats.extend(czlabels)
    bigcats.extend(enlabels)
    print("****Multilingual classification baseline*************")
    singleLangClassificationWithoutVectorizer(bigfeats, bigcats)


if __name__ == "__main__":
    main()
