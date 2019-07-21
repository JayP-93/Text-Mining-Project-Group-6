# Purpose: Build a scorer with POS N-grams. Use it on another language.

import pprint
import os
import collections
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer  # to replace NaN with mean values.
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, \
    RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import LinearSVC

from scipy.stats import spearmanr, pearsonr

import language_check
import logging

seed = 1234


def makePOSsentences(conllufilepath):
    """
        go through all lines and read each line of the table which represents information of the text.
        we read the fourth column which contains POS tag of a word in the line of table.

    :param conllufilepath:  path to the parsed file
    :return:  a string which contains the sequence of sentences. words of a sentence represent
    its Part-Of-Speech tag(Example: Stadt - NOUN, habe - VERB).
    """
    fh = open(conllufilepath, encoding="utf-8")
    everything_POS = []
    pos_sentence = []
    sent_id = 0
    for line in fh:
        if line == "\n":
            pos_string = " ".join(pos_sentence) + "\n"
            everything_POS.append(pos_string)
            pos_sentence = []
            sent_id = sent_id + 1
        elif not line.startswith("#"):
            pos_tag = line.split("\t")[3]
            pos_sentence.append(pos_tag)
    fh.close()
    return " ".join(everything_POS)  # Returns a string which contains one sentence as POS tag sequence per line



def makeTextOnly(conllufilepath):
    """
        go through all lines and read each line of the table which represents information of the text.
        we get a sequence of sentences of the plain text(each sentence per new line).
    :param conllufilepath: path to the parsed file
    :return: a string which contains the sequence of sentences

    :quetions: the idea of using sent_id
    """
    fh = open(conllufilepath, encoding="utf-8")
    allText = []
    this_sentence = []
    sent_id = 0
    for line in fh:
        if line == "\n":
            word_string = " ".join(this_sentence) + "\n"
            allText.append(word_string)
            this_sentence = []
            sent_id = sent_id + 1
        elif not line.startswith("#"):
            word = line.split("\t")[1]
            this_sentence.append(word)
    fh.close()
    return " ".join(allText)  # Returns a string which contains one sentence as POS tag sequence per line


def makeDepRelSentences(conllufilepath):
    """
        convert a sentence into this form: nmod_NN_PRON, dobj_VB_NN etc. i.e., each word is replaced by a dependency trigram of that form.
        a triplet consisting of dependency relation, POS tag of the dependent, POS tag of the head
        So full text will look like this instead of a series of words or POS tags:
        root_PRON_ROOT nsubj_NOUN_PRON case_ADP_PROPN det_DET_PROPN nmod_PROPN_NOUN
        case_ADP_NOUN det_DET_NOUN nummod_NUM_NOUN obl_NOUN_VERB root_VERB_ROOT case_ADP_NOUN det_DET_NOUN obl_NOUN_VERB appos_PROPN_NOUN flat_PROPN_PROPN case_ADP_NOUN obl_NOUN_VERB cc_CCONJ_PART conj_PART_PROPN punct_PUNCT_VERB
    :param conllufilepath: path to the parsed file
    :return: a string which contains the sequence of sentences. words of a sentence represent
    """
    fh = open(conllufilepath, encoding="utf-8")
    wanted_features = []
    deprels_sentence = []
    sent_id = 0
    head_ids_sentence = []
    pos_tags_sentence = []
    wanted_sentence_form = []
    id_dict = {}  # Key: Index, Value: Word or POS depending on what dependency trigram we need. I am taking POS for now.
    id_dict['0'] = "ROOT"
    for line in fh:
        if line == "\n":
            for rel in deprels_sentence:
                wanted = rel + "_" + pos_tags_sentence[deprels_sentence.index(rel)] + "_" + id_dict[
                    head_ids_sentence[deprels_sentence.index(rel)]]
                wanted_sentence_form.append(wanted)

                # Trigrams of the form case_ADP_PROPN, flat_PROPN_PROPN etc.
            wanted_features.append(" ".join(wanted_sentence_form) + "\n")
            deprels_sentence = []
            pos_tags_sentence = []
            head_ids_sentence = []
            wanted_sentence_form = []
            sent_id = sent_id + 1
            id_dict = {}
            id_dict['0'] = "root"  # LOWERCASING. Some problem with case of features in vectorizer.

        elif not line.startswith("#") and "-" not in line.split("\t")[0]:
            fields = line.split("\t")
            # read POS tag of a word
            pos_tag = fields[3]
            deprels_sentence.append(fields[7])
            id_dict[fields[0]] = pos_tag
            pos_tags_sentence.append(pos_tag)
            head_ids_sentence.append(fields[6])
    fh.close()
    return " ".join(wanted_features)


"""
As described in Lu, 2010: http://onlinelibrary.wiley.com/doi/10.1111/j.1540-4781.2011.01232_1.x/epdf
Lexical words (N_lex: all open-class category words in UD (ADJ, ADV, INTJ, NOUN, PROPN, VERB)
All words (N)
Lex.Density = N_lex/N
Lex. Variation = Uniq_Lex/N_Lex
Type-Token Ratio = Uniq_words/N
Verb Variation = Uniq_Verb/N_verb
Noun Variation
ADJ variation
ADV variation
Modifier variation
"""


def getLexFeatures(conllufilepath, lang, err):
    """
        get the scoring of lexical features of the text, for example number of nouns, verbs, spelling mistakes,
        total lexical words, length of script nad so on.
    :param conllufilepath: the path to the directory where all texts corresponding to the specific language are kept
    :param lang: abbreviation of a language
    :param err: true - extract error features, false - otherwise, do not do that(Example:no exiting tool for grammar check)
    :return: an array where each line represents scoring features of a text such as [length of script,
    the mean lenght of sentences, Type-Token Ratio, lexical diversity, lexical variation,
    the frequency occurrence of verbs, the frequency occurrence of nouns, the frequency occurrence of adjectives,
    the frequency occurrence of adverbs,the frequency occurrence of modifier, total number of errors, total numbers of spelling mistakes]
    """
    fh = open(conllufilepath, encoding="utf-8")
    ndw = []  # To get number of distinct words
    ndn = []  # To get number of distinct nouns - includes propn
    ndv = []  # To get number of distinct verbs
    ndadj = []  # To get number of distinct adjectives
    ndadv = []  # To get number of distinct adverbs
    ndint = []  # To get number of distinct interjecion?
    numN = 0.0  # Including PROPN->INCL PROPN
    numV = 0.0
    numI = 0.0  # INCJ -> including J?
    numADJ = 0.0
    numADV = 0.0
    numIntj = 0.0  # does not use
    total = 0.0  # total number including words, commends and an empty lines
    numSent = 0.0
    for line in fh:
        if not line == "\n" and not line.startswith("#"):
            fields = line.split("\t")
            word = fields[1]
            pos_tag = fields[3]
            if word.isalpha():
                if not word in ndw:
                    ndw.append(
                        word)  # add to an array of distinct words, and then add to an array its coressponding Part-Of-Speech
                if pos_tag == "NOUN" or pos_tag == "PROPN":
                    numN = numN + 1
                    if not word in ndn:
                        ndn.append(word)
                elif pos_tag == "ADJ":
                    numADJ = numADJ + 1
                    if not word in ndadj:
                        ndadj.append(word)
                elif pos_tag == "ADV":
                    numADV = numADV + 1
                    if not word in ndadv:
                        ndadv.append(word)
                elif pos_tag == "VERB":
                    numV = numV + 1
                    if not word in ndv:
                        ndv.append(word)
                elif pos_tag == "INTJ":
                    numI = numI + 1
                    if not word in ndint:
                        ndint.append(word)
        elif line == "\n":
            numSent = numSent + 1
        total = total + 1
    if err:  # get the error features of the text  if there is an exception that we ignore this text and set up null values
        try:
            error_features = getErrorFeatures(conllufilepath, lang)
        except:
            print("Ignoring file:", conllufilepath)
            error_features = [0, 0]
    else:
        error_features = ['NA', 'NA']

    nlex = float(numN + numV + numADJ + numADV + numI)  # Total Lexical words i.e., tokens
    dlex = float(len(ndn) + len(ndv) + len(ndadj) + len(ndadv) + len(ndint))  # Distinct Lexical words i.e., types
    # Scriptlen, Mean Sent Len, TTR, LexD, LexVar, VVar, NVar, AdjVar, AdvVar, ModVar, Total_Errors, Total Spelling errors
    result = [total, round(total / numSent, 2), round(len(ndw) / total, 2), round(nlex / total, 2),
              round(dlex / nlex, 2), round(len(ndv) / nlex, 2), round(len(ndn) / nlex, 2),
              round(len(ndadj) / nlex, 2), round(len(ndadv) / nlex, 2), round((len(ndadj) + len(ndadv)) / nlex, 2),
              error_features[0], error_features[1]]
    if not err:  # remove last two features - they are error features which are NA for cz
        return result[:-2]
    else:
        return result


"""
Num. Errors. NumSpellErrors
May be other error based features can be added later.
"""


def getErrorFeatures(conllufilepath, lang):
    """
        check the text for error and count them. if there is an exception, then we ignore this text
    :param conllufilepath: text to be read
    :param lang: language of the text
    :return:  array consists of two columns . the first column - number of error including grammatical, punctuation
     and spelling mistakes, the second column - spelling errors
    """
    numerr = 0
    numspellerr = 0
    try:
        checker = language_check.LanguageTool(lang)
        text = makeTextOnly(conllufilepath)
        matches = checker.check(text)
        for match in matches:
            if not match.locqualityissuetype == "whitespace":
                numerr = numerr + 1
                if match.locqualityissuetype == "typographical" or match.locqualityissuetype == "misspelling":
                    numspellerr = numspellerr + 1
    except:
        print("Ignoring this text: ", conllufilepath)

    return [numerr, numspellerr]


def getScoringFeatures(dirpath, lang, err):
    """
        read all text from files from the given directory
        call the method getLexFeatures to get scoring features of a text

    :param dirpath: the path to the directory where all texts corresponding to the specific language are kept
    :param lang: abbreviation of a language
    :param err: true - to extract error features, false - otherwise, do not do that(fpr example, no exiting tool for grammar check)
    :return: two different arrays. the first array contains files.
    the second array consist of scoring features of a text respectively
    """
    files = os.listdir(dirpath)
    fileslist = []
    featureslist = []
    for filename in files:
        if filename.endswith(".txt"):
            features_for_this_file = getLexFeatures(os.path.join(dirpath, filename), lang, err)
            fileslist.append(filename)
            featureslist.append(features_for_this_file)
    return fileslist, featureslist


def getLangData(dirpath, option):
    """
        we read all files by calling function os.listdir()
        then we call a function with respect to an option for each text
        afterward we get a spring that contains features of a text
    :param dirpath: the path to the directory where all texts corresponding to the specific language are kept
    :param option: which feature to extract from the text. options:word n-grams,POS n-grams, dependency representations.
    word option is default option
    :return: two arrays. the first array is the array of all files,
    the second array is the array of strings which represent features of corresponding text respectively
    """
    files = os.listdir(dirpath)
    fileslist = []
    posversionslist = []
    for filename in files:
        if filename.endswith(".txt"):
            if option == "pos":
                pos_version_of_file = makePOSsentences(
                    os.path.join(dirpath, filename))  # DO THIS TO GET POS N-GRAM FEATURES later
            elif option == "dep":
                pos_version_of_file = makeDepRelSentences(
                    os.path.join(dirpath, filename))  # DO THIS TO GET DEP-TRIAD N-gram features later
            else:
                pos_version_of_file = makeTextOnly(
                    os.path.join(dirpath, filename))  # DO THIS TO GET Word N-gram features later
            fileslist.append(filename)
            posversionslist.append(pos_version_of_file)
    return fileslist, posversionslist


def getcatlist(filenameslist):
    """
        Get categories (CEFR levels) from filenames -Classification
    :param filenameslist:  list of all files
    :return: the array containing CEFR levels of texts
    """
    result = []
    for name in filenameslist:
        result.append(name.split(".txt")[0].split("_")[-1])
    return result


def getlangslist(filenameslist):
    """
     Get langs list(CEFR levels) from filenames - to use in megadataset classification
    :param filenameslist:  list of all files
    :return: the array containing CEFR levels of texts
    """
    result = []
    for name in filenameslist:
        if "_DE_" in name:
            result.append("de")
        elif "_IT_" in name:
            result.append("it")
        else:
            result.append("cz")
    return result


def getnumlist(filenameslist):
    """
        Get numbers representing (CEFR levels) from filenames - Regression
    :param filenameslist:  list of all files
    :return: the array containing CEFR levels of texts
    """
    result = []
    mapping = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
    for name in filenameslist:
        result.append(mapping[name.split(".txt")[0].split("_")[-1]])
    return result


def regEval(predicted, actual):
    """
        do regression evation based on Mean Absolute Error, Pearson correlation coefficient,
        Spearman's rank correlation coefficient, Root Mean Squared Logarithmic Error
    :param predicted: predicted data(CEFR levels) of text
    :param actual: calculated (CEFR levels) data by POS
    :return: dictionary consists of different ways of regression evaluations and their coefficient or loss
    """
    n = len(predicted)
    MAE = mean_absolute_error(actual, predicted)
    pearson = pearsonr(actual, predicted)  # Pearson correlation coefficient
    spearman = spearmanr(actual, predicted)  # Spearman's rank correlation coefficient
    rmsle = np.sqrt((1 / n) * sum(
        np.square(np.log10(predicted + 1) - (np.log10(actual) + 1))))  # Root Mean Squared Logarithmic Error
    return {"MAE": MAE, "rmlse": rmsle, "pearson": pearson, "spearman": spearman}


def train_onelang_classification(train_labels, train_data, labelascat=False, langslist=None):
    """
        training on one language data with random forest Classifier
    :param train_labels: labels of data(CEFR levels)
    :param train_data: data to be train
    :param labelascat:true, false (to indicate whether to add label as a categorical feature)
    :param langslist: the list of all textes of all languages
    """
    uni_to_tri_vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                            ngram_range=(1, 5), min_df=10)
    vectorizers = [uni_to_tri_vectorizer]

    # Create a random forest Classifier.
    classifiers = [RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=seed),
                   LinearSVC(class_weight="balanced", random_state=seed), LogisticRegression(class_weight="balanced",
                                                                                             random_state=seed)]  # Add more.GradientBoostingClassifier(),

    # Provides train/test indices to split data in train/test sets, use 1/10 for traning
    k_fold = StratifiedKFold(10, random_state=seed)
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))

            # to covert a collection of text documents to term-document matrix. for each text we have vector,
            # the element represents features(words or n-grams) and the number of their occurrence
            train_vector = vectorizer.fit_transform(train_data).toarray()
            print(len(train_vector[0]))

            if labelascat and len(langslist) > 1:
                # extent training vector by adding labels vectors
                train_vector = enhance_features_withcat(train_vector, language=None, langslist=langslist)
            print(len(train_vector[0]))

            # print(vectorizer.get_feature_names()) #To see what features were selected.
            # to estimate the accuracy of a classification by splitting the data, fitting a model
            # and computing the score 10 consecutive times
            cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold,
                                        n_jobs=1)  # Evaluate a score by cross-validation

            # for each element in an array, the prediction that was obtained for that element when it was in the test set
            predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold,
                                          n_jobs=1)  # Generate cross-validated estimates for each input data point
            print(cross_val)

            # calculate  a weighted average of the precision and recall
            print(sum(cross_val) / float(len(cross_val)), f1_score(train_labels, predicted, average='weighted'))

            # print(vectorizer.get_feature_names())
            print(confusion_matrix(train_labels, predicted, labels=["A1", "A2", "B1", "B2", "C1", "C2"]))
            # print(predicted)
    print("SAME LANG EVAL DONE FOR THIS LANG")


def combine_features(train_labels, train_sparse, train_dense):
    """
    Combine features like this: get probability distribution over categories with n-gram features.
    Use that distribution as a feature set concatenated with the domain features - one way to combine sparse and dense
    feature groups.
    Just testing this approach here.
    :param train_labels:  labels of data(CEFR levels)
    :param train_sparse: a set of features(word n-grams,POS n-grams, dependency representations)
    that may have values of zero
    :param train_dense: a set of scoring features that have only non-zero values
    """
    # Provides train/test indices to split data in train/test sets, use 1/10 for traning
    k_fold = StratifiedKFold(10, random_state=seed)
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 ngram_range=(1, 3), min_df=10, max_features=2000)
    train_vector = vectorizer.fit_transform(train_sparse).toarray()
    classifier = RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=seed)
    # cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
    # print("Old CV score with sparse features", str(sum(cross_val)/float(len(cross_val))))
    # predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold)
    # print(f1_score(train_labels,predicted,average='weighted'))

    # Get probability distribution for classes.
    predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold, method="predict_proba")

    # Use those probabilities as the new feature set.
    new_features = []
    for i in range(0, len(predicted)):
        temp = list(predicted[i]) + list(train_dense[i])
        new_features.append(temp)

    # predict new features
    new_predicted = cross_val_predict(classifier, new_features, train_labels, cv=k_fold)
    cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)

    # calculate accuracy
    print("Acc: ", str(sum(cross_val) / float(len(cross_val))))

    # calculate a weighted average of the precision and recall
    print("F1: ", str(f1_score(train_labels, new_predicted, average='weighted')))


"""
Single language, regression with 10 fold CV
"""


def train_onelang_regression(train_scores, train_data):
    """
        training on one language data with different regression model(Linear Regression,Gradient Boosting Regressor and so on )
    :param train_scores: the score (representing CEFR level)of given text
    :param train_data: data to be train for different regressors
    """
    uni_to_tri_vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                            ngram_range=(1, 5),
                                            min_df=10)  # can specify max_features but dataset seems small enough
    vectorizers = [uni_to_tri_vectorizer]
    regressors = [LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]
    # Provides train/test indices to split data in train/test sets, use 1/10 for traning
    k_fold = StratifiedKFold(10, random_state=seed)
    for vectorizer in vectorizers:
        for regressor in regressors:
            # to covert a collection of text documents to term-document matrix. for each text we have vector,
            # the element represents features(words or n-grams) and the number of their occurrence (tokenization)
            train_vector = vectorizer.fit_transform(train_data).toarray()
            print("Printing results for: " + str(regressor) + str(vectorizer))

            # to estimate the accuracy of a classification for each fold
            cross_val = cross_val_score(regressor, train_vector, train_scores, cv=k_fold, n_jobs=1)

            # do prediction
            predicted = cross_val_predict(regressor, train_vector, train_scores, cv=k_fold)
            predicted[predicted < 0] = 0
            n = len(predicted)

            # evaluate the correctness of prediction based different regression evaluation metrics
            print(regEval(predicted, train_scores))
    print("SAME LANG EVAL DONE")


def cross_lang_testing_classification(train_labels, train_data, test_labels, test_data):
    """
        train on one language and test on another, classification
        use pipeline because these are two transformations at least.
        first one when training the model and again on any new data you want to predict on.
    :param train_labels: labels of texts representing CEFR levels to train
    :param train_data: text to be trained
    :param test_labels: labels of texts representing CEFR levels to text
    :param test_data:text to be tested
    """
    # for prefering data for classification
    uni_to_tri_vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                            ngram_range=(1, 5), min_df=10)  # , max_features = 2000
    vectorizers = [uni_to_tri_vectorizer]
    classifiers = [RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=seed),
                   LinearSVC(class_weight="balanced", random_state=seed), LogisticRegression(class_weight="balanced",
                                                                                             random_state=seed)]
    # LinearSVC(), RandomForestClassifier(), RandomForestClassifier(class_weight="balanced"), GradientBoostingClassifier()]
    # #Side note: gradient boosting needs a dense array. Testing fails for that. Should modifiy the pipeline later to account for this.
    # Check this discussion for handling the sparseness issue: https://stackoverflow.com/questions/28384680/scikit-learns-pipeline-a-sparse-matrix-was-passed-but-dense-data-is-required
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))

            # combines the vectorizer created above with a classifier.
            text_clf = Pipeline([('vect', vectorizer), ('clf', classifier)])

            # firstly do tokenization and then train classifier
            text_clf.fit(train_data, train_labels)
            # print(vectorizer.get_feature_names())

            # firstly do tokenization and then predict new data
            predicted = text_clf.predict(test_data)
            # print(vectorizer.get_feature_names())
            print(np.mean(predicted == test_labels, dtype=float))
            print(confusion_matrix(test_labels, predicted, labels=["A1", "A2", "B1", "B2", "C1", "C2"]))
            print("CROSS LANG EVAL DONE. F1score: ")
            # calculate a weighted average of the precision and recall
            print(f1_score(test_labels, predicted, average='weighted'))


"""
Note: XGBoost classifier has some issue with retaining feature names between train and test data properly. This is resulting in error while doing cross language classification.
Strangely, I did not encounter this issue with POS trigrams. Only encountering with dependency features.
Seems to be a known issue: https://github.com/dmlc/xgboost/issues/2334
"""


# train on one language and test on another, classification
def cross_lang_testing_regression(train_scores, train_data, test_scores, test_data):
    """
        train on one language and test on another, classification
        use pipeline because these are two transformations at least.
        first one when training the model and again on any new data you want to predict on.

    :param train_scores: scores of texts representing CEFR levels to train
    :param train_data: text to be trained
    :param test_scores: scores of texts representing CEFR levels to text
    :param test_data:text to be tested
    """
    uni_to_tri_vectorizer = CountVectorizer(analyzer="char", tokenizer=None, preprocessor=None, stop_words=None,
                                            ngram_range=(1, 10), min_df=10, max_features=10000)
    vectorizers = [uni_to_tri_vectorizer]
    regressors = [
        RandomForestRegressor()]  # linear_model.LinearRegression(),  - seems to be doing badly for cross-lang.
    for vectorizer in vectorizers:
        for regressor in regressors:
            train_vector = vectorizer.fit_transform(train_data).toarray()
            print("Printing results for: " + str(regressor) + str(vectorizer))

            # combines the vectorizer created above with a classifier.
            text_clf = Pipeline([('vect', vectorizer), ('clf', regressor)])

            # firstly do tokenization and then train classifier
            text_clf.fit(train_data, train_scores)

            # firstly do tokenization and then predict new data
            predicted = text_clf.predict(test_data)
            predicted[predicted < 0] = 0
            n = len(predicted)

            # evaluate the correctness of prediction based different regression evaluation metrics
            print("RMSLE: ", np.sqrt((1 / n) * sum(np.square(np.log10(predicted + 1) - (np.log10(test_scores) + 1)))))
            print("MAE: ", mean_absolute_error(test_scores, predicted))
            print("Pearson: ", pearsonr(test_scores, predicted))
            print("Spearman: ", spearmanr(test_scores, predicted))


# Single language, 10 fold cv for domain features - i.e., non n-gram features.
def singleLangClassificationWithoutVectorizer(train_vector, train_labels):  # test_vector,test_labels):
    """
         train and test on one language, classification
    :param train_vector: we use the matrix where Y axis - texts, X - scoring features to train cassifier
    :param train_labels: labels of texts representing CEFR levels to train
    """
    # Provides train/test indices to split data in train/test sets, use 1/10 for training
    k_fold = StratifiedKFold(10, random_state=seed)
    classifiers = [RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=seed),
                   LinearSVC(class_weight="balanced", random_state=seed),
                   LogisticRegression(class_weight="balanced", random_state=seed)]
    # classifiers = [MLPClassifier(max_iter=500)]
    # RandomForestClassifer(), GradientBoostClassifier()
    # Not useful: SVC with kernels - poly, sigmoid, rbf.
    for classifier in classifiers:
        print(classifier)

        # to estimate the accuracy of the classifier on the  dataset by splitting the data, fitting a model
        # and computing the score 10 consecutive times
        cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)

        # predict new features
        predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold)

        print(cross_val)
        print(sum(cross_val) / float(len(cross_val)))
        print(confusion_matrix(train_labels, predicted))
        # calculate a weighted average of the precision and recall
        print(f1_score(train_labels, predicted, average='macro'))


# cross lingual classification evaluation for non ngram features
def crossLangClassificationWithoutVectorizer(train_vector, train_labels, test_vector, test_labels):
    """
    train classifiers on one language and test on another language
    :param train_vector: we use the matrix where Y axis - texts, X - scoring features for training
    :param train_labels: labels of texts representing CEFR levels for training
    :param test_vector: we use the matrix where Y axis - texts, X - scoring features for testing
    :param test_labels: labels of texts representing CEFR levels for testing
    """
    print("CROSS LANG EVAL")
    classifiers = [RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=seed),
                   LinearSVC(class_weight="balanced", random_state=seed),
                   LogisticRegression(class_weight="balanced", random_state=seed)]
    for classifier in classifiers:
        # Train classifier on train set
        classifier.fit(train_vector, train_labels)
        # Test classifier on test set and compute mean matches, confusion matrix and F1 score
        predicted = classifier.predict(test_vector)
        print(np.mean(predicted == test_labels, dtype=float))
        print(confusion_matrix(test_labels, predicted))
        print(f1_score(test_labels, predicted, average='weighted'))


# cross lingual regression evaluation for non ngram features
def crossLangRegressionWithoutVectorizer(train_vector, train_scores, test_vector, test_scores):
    """
    train regressors on one language and test on another language
    :param train_vector: we use the matrix where Y axis - texts, X - scoring features for training
    :param train_scores: scores of texts representing CEFR levels for training
    :param test_vector: we use the matrix where Y axis - texts, X - scoring features for testing
    :param test_scores: scores of texts representing CEFR levels for testing
    """
    print("CROSS LANG EVAL")
    regressors = [RandomForestRegressor()]
    k_fold = StratifiedKFold(10, random_state=seed)
    for regressor in regressors:
        cross_val = cross_val_score(regressor, train_vector, train_scores, cv=k_fold, n_jobs=1)
        predicted = cross_val_predict(regressor, train_vector, train_scores, cv=k_fold)
        predicted[predicted < 0] = 0
        print("Cross Val Results: ")
        print(regEval(predicted, train_scores))
        regressor.fit(train_vector, train_scores)
        predicted = regressor.predict(test_vector)
        predicted[predicted < 0] = 0
        print("Test data Results: ")
        print(regEval(predicted, test_scores))


# add label features as one hot vector. de - 1 0 0, it - 0 1 0, cz - 0 0 1 as sklearn has issues with combination of cat and num features.
def enhance_features_withcat(features, language=None, langslist=None):
    """
    Adds label features to the features vector as one hot vector defined as de - 1 0 0, it - 0 1 0, cz - 0 0 1
    :param features: Existing features vector to be extended with the label features
    :param language: The language label to be added
    :param langslist: The language labels to be added
    :return: The extended features vector
    """
    addition = {'de': [1, 0, 0], 'it': [0, 1, 0], 'cz': [0, 0, 1]}
    if language:
        for i in range(0, len(features)):
            features[i].extend(addition[language])
        return features
    if langslist:
        features = np.ndarray.tolist(features)
        for i in range(0, len(features)):
            features[i].extend(addition[langslist[i]])
        return features


"""
Goal: combine all languages data into one big model
setting options: pos, dep, domain, word
labelascat = true, false (to indicate whether to add label as a categorical feature)
"""


def do_mega_multilingual_model_all_features(lang1path, lang1, lang2path, lang2, lang3path, lang3, modelas, setting,
                                            labelascat):
    """
    Prepare dataset of multiple languages and train and test classifiers on it
    :param lang1path: Directory path to the first language
    :param lang1: Abbreviation of the first language (label)
    :param lang2path: Directory path to the second language
    :param lang2: Abbreviation of the second language (label)
    :param lang3path: Directory path of the third language
    :param lang3: Abbreviation of the third language (label)
    :param modelas: not used
    :param setting: which setting to use (to determine features) possible values: pos, dep, domain, word
    :param labelascat: boolean value to indicate whether to add label as a categorical feature or not
    """

    print("Doing: take all data as if it belongs to one large dataset, and do classification")
    if not setting == "domain":
        lang1files, lang1features = getLangData(lang1path, setting)
        lang1labels = getcatlist(lang1files)
        lang2files, lang2features = getLangData(lang2path, setting)
        lang2labels = getcatlist(lang2files)
        lang3files, lang3features = getLangData(lang3path, setting)
        lang3labels = getcatlist(lang3files)

    else:  # i.e., domain features only.
        lang1files, lang1features = getScoringFeatures(lang1path, lang1, False)
        lang1labels = getcatlist(lang1files)
        lang2files, lang2features = getScoringFeatures(lang2path, lang2, False)
        lang2labels = getcatlist(lang2files)
        lang3files, lang3features = getScoringFeatures(lang3path, lang3, False)
        lang3labels = getcatlist(lang3files)

    # Add all the languages to a single dataset
    megalabels = lang1labels + lang2labels + lang3labels
    megalangs = getlangslist(lang1files) + getlangslist(lang2files) + getlangslist(lang3files)

    # Add label features
    if labelascat and setting == "domain":
        megadata = enhance_features_withcat(lang1features, "de") + enhance_features_withcat(lang2features, "it") \
                   + enhance_features_withcat(lang3features, "cz")
    else:
        megadata = lang1features + lang2features + lang3features

    print("Mega classification for: ", setting, " features")

    print(len(megalabels), len(megadata), len(megalangs), len(megadata[0]))

    print("Distribution of labels: ")
    print(collections.Counter(megalabels))

    # Train and test classifiers
    if setting == "domain":
        singleLangClassificationWithoutVectorizer(megadata, megalabels)
    else:
        train_onelang_classification(megalabels, megadata, labelascat, megalangs)


"""
this function does cross language evaluation.
takes a language data directory path, and lang code for both source and target languages. 
gets all features (no domain features for cz), and prints the results with those.
lang codes: de, it, cz (lower case)
modelas: "class" for classification, "regr" for regression
"""


def do_cross_lang_all_features(sourcelangdirpath, sourcelang, modelas, targetlangdirpath, targetlang):
    """
    Prepare dataset of 2 languages: one to train on and one to test on. Then do training and testing.
    :param sourcelangdirpath: Directory path to the train language
    :param sourcelang: Abbreviation of the train language (label)
    :param modelas: Flag to indicate what kind of learners (classifiers or regressors) to use. Values: class, regr (regr not implemented yet)
    :param targetlangdirpath: Directory path to the test language
    :param targetlang: Abbreviation of the test language (label)
    """
    # Read source language data
    sourcelangfiles, sourcelangposngrams = getLangData(sourcelangdirpath, "pos")
    sourcelangfiles, sourcelangdepngrams = getLangData(sourcelangdirpath, "dep")
    # Read target language data
    targetlangfiles, targetlangposngrams = getLangData(targetlangdirpath, "pos")
    targetlangfiles, targetlangdepngrams = getLangData(targetlangdirpath, "dep")
    # Get label info
    sourcelanglabels = getcatlist(sourcelangfiles)
    targetlanglabels = getcatlist(targetlangfiles)

    # Because cz has no error check so error features cannot be collected
    if "cz" not in [sourcelang, targetlang]:
        sourcelangfiles, sourcelangdomain = getScoringFeatures(sourcelangdirpath, sourcelang, True)
        targetlangfiles, targetlangdomain = getScoringFeatures(targetlangdirpath, targetlang, True)
    else:
        sourcelangfiles, sourcelangdomain = getScoringFeatures(sourcelangdirpath, sourcelang, False)
        targetlangfiles, targetlangdomain = getScoringFeatures(targetlangdirpath, targetlang, False)

    diff_labels = set(targetlanglabels) - set(sourcelanglabels)
    if diff_labels:
        indices = [i for i, x in enumerate(targetlanglabels) if x in diff_labels]
        targetlangposngrams = [i for j, i in enumerate(targetlangposngrams) if j not in indices]
        targetlangdepngrams = [i for j, i in enumerate(targetlangdepngrams) if j not in indices]
        targetlanglabels = [x for x in targetlanglabels if x not in diff_labels]
        targetlangdomain = [i for j, i in enumerate(targetlangdomain) if j not in indices]

        # if targetlang == "it": #Those two files where langtool throws error
        #   mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        #   mean_imputer = mean_imputer.fit(targetlangdomain)
        #   imputed_df = mean_imputer.transform(targetlangdomain)
        #   targetlangdomain = imputed_df
        #   print("Modified domain feature vector for Italian")
        # TODO: it can be sourcelang too! I am ignoring that for now.

    if modelas == "class":
        print("Printing cross-corpus classification evaluation results: ")
        print("*******", "\n", "Setting - Train with: ", sourcelang, " Test with: ", targetlang, " ******", "\n")
        print("Features: pos")
        cross_lang_testing_classification(sourcelanglabels, sourcelangposngrams, targetlanglabels, targetlangposngrams)
        print("Features: dep")
        cross_lang_testing_classification(sourcelanglabels, sourcelangdepngrams, targetlanglabels, targetlangdepngrams)
        print("Features: domain")
        crossLangClassificationWithoutVectorizer(sourcelangdomain, sourcelanglabels, targetlangdomain, targetlanglabels)
    if modelas == "regr":
        print("Did not add for regression yet")


"""
this function takes a language data directory path, and lang code, 
gets all features, and prints the results with those.
lang codes: de, it, cz (lower case)
modelas: "class" for classification, "regr" for regression
"""


def do_single_lang_all_features(langdirpath, lang, modelas):
    """
    Prepare dataset from a single language with different features vectors (word, post, dep). Then do training and testing.
    :param langdirpath: Directory path to the train/test language
    :param lang: Abbreviation of the train/test language (label)
    :param modelas: modelas: Flag to indicate what kind of learners (classifiers or regressors) to use. Values: class, regr
    """
    langfiles, langwordngrams = getLangData(langdirpath, "word")
    langfiles, langposngrams = getLangData(langdirpath, "pos")
    langfiles, langdepngrams = getLangData(langdirpath, "dep")
    if not lang == "cz":
        langfiles, langdomain = getScoringFeatures(langdirpath, lang, True)
    else:
        langfiles, langdomain = getScoringFeatures(langdirpath, lang, False)

    print("Extracted all features: ")
    langlabels = getcatlist(langfiles)
    langscores = getnumlist(langfiles)

    # if lang == "it": #Those two files where langtool throws error
    #    mean_imputer = Imputer(missing_values='NA', strategy='mean', axis=0)
    #    mean_imputer = mean_imputer.fit(langdomain)
    #    imputed_df = mean_imputer.transform(langdomain)
    #    langdomain = imputed_df
    #    print("Modified domain feature vector for Italian")

    print("Printing class statistics")
    print(collections.Counter(langlabels))

    if modelas == "class":
        print("With Word ngrams:", "\n", "******")
        train_onelang_classification(langlabels, langwordngrams)
        print("With POS ngrams: ", "\n", "******")
        train_onelang_classification(langlabels, langposngrams)
        print("Dep ngrams: ", "\n", "******")
        train_onelang_classification(langlabels, langdepngrams)
        print("Domain features: ", "\n", "******")
        singleLangClassificationWithoutVectorizer(langdomain, langlabels)

        print("Combined feature rep: wordngrams + domain")
        combine_features(langlabels, langwordngrams, langdomain)
        print("Combined feature rep: posngrams + domain")
        combine_features(langlabels, langposngrams, langdomain)
        print("Combined feature rep: depngrams + domain")
        combine_features(langlabels, langdepngrams, langdomain)
        # TODO
        # print("ALL COMBINED")

        """
       defiles,dedense = getScoringFeatures(dedirpath, "de", True)
       defiles,desparse = getLangData(dedirpath)
       delabels = getcatlist(defiles)
       combine_features(delabels,desparse,dedense)
       """
    elif modelas == "regr":
        print("With Word ngrams:", "\n", "******")
        train_onelang_regression(langscores, langwordngrams)
        print("With POS ngrams: ", "\n", "******")
        train_onelang_regression(langscores, langposngrams)
        print("Dep ngrams: ", "\n", "******")
        train_onelang_regression(langscores, langwordngrams)
        # TODO: singleLangRegressionWithoutVectorizer function.
        # print("Domain features: ", "\n", "******")
        # singleLangRegressionWithoutVectorizer(langdomain,langlabels)


def main():
    # TODO(JayP): adapt this when you want to run it!
    itdirpath = "..\\Datasets\\IT-Parsed"
    dedirpath = "..\\Datasets\\DE-Parsed"
    czdirpath = "..\\Datasets\\CZ-Parsed"
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Script Begin')

    # logging.info('Starting with single language --> DE')
    # logging.info('Starting do_single_lang_all_features(dedirpath, "de", "class")')
    # do_single_lang_all_features(dedirpath, "de", "class")
    # logging.info('Finished do_single_lang_all_features(dedirpath, "de", "class")')
    #
    # logging.info('Starting do_single_lang_all_features(dedirpath, "de", "regr")')
    # do_single_lang_all_features(dedirpath, "de", "regr")
    # logging.info('Finished do_single_lang_all_features(dedirpath, "de", "regr")')
    # logging.info('Finished with single language --> DE')
    #
    # logging.info('Starting with single language --> IT')
    # logging.info('Starting do_single_lang_all_features(itdirpath, "it", "class")')
    # do_single_lang_all_features(itdirpath, "it", "class")
    # logging.info('Finished do_single_lang_all_features(itdirpath, "it", "class")')
    #
    # logging.info('Starting do_single_lang_all_features(itdirpath, "it", "regr")')
    # do_single_lang_all_features(itdirpath, "it", "regr")
    # logging.info('Finished do_single_lang_all_features(itdirpath, "it", "regr")')
    # logging.info('Finished with single language --> IT')
    #
    # logging.info('Starting with single language --> CZ')
    # logging.info('Starting do_single_lang_all_features(czdirpath, "cz", "class")')
    # do_single_lang_all_features(czdirpath, "cz", "class")
    # logging.info('Finished do_single_lang_all_features(czdirpath, "cz", "class")')
    #
    # logging.info('Starting do_single_lang_all_features(czdirpath, "cz", "regr")')
    # do_single_lang_all_features(czdirpath, "cz", "regr")
    # logging.info('Finished do_single_lang_all_features(czdirpath, "cz", "regr")')
    # logging.info('Finished with single language --> cz')

    # logging.info("Starting with concatenate label features --> True")
    # logging.info(
    #     'Starting do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "word", True)')
    # do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "word", True)
    # logging.info(
    #     'Finished do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "word", True)')
    # logging.info(
    #     'Starting do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "pos", True)')
    # do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "pos", True)
    # logging.info(
    #     'Finished do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "pos", True)')
    #
    # logging.info(
    #     'Starting do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "dep", True)')
    # do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "dep", True)
    # logging.info(
    #     'Finished do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "dep", True)')
    #
    # logging.info(
    #     'Starting do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "domain", True)')
    # do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "domain", True)
    # logging.info(
    #     'Finished do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "domain", True)')
    # logging.info("Finished with concatenate label features --> True")
    #
    # logging.info("Starting with concatenate label features --> False")
    logging.info(
         'Starting do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "word", False)')
    do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "word", False)
    logging.info(
        'Finished do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "word", False)')
    # logging.info(
    #     'Starting do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "pos", False)')
    # do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "pos", False)
    # logging.info(
    #     'Finished do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "pos", False)')
    #
    # logging.info(
    #     'Starting do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "dep", False)')
    # do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "dep", False)
    # logging.info(
    #     'Finished do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "dep", False)')
    #
    # logging.info(
    #     'Starting do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "domain", False)')
    # do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "domain", False)
    # logging.info(
    #     'Finished do_mega_multilingual_model_all_features(dedirpath, "de", itdirpath, "it", czdirpath, "cz", "class", "domain", False)')
    # logging.info("Finished with concatenate label features --> False")

    logging.info("Starting cross language with DE as base language")
    logging.info('Starting cross language from DE to IT do_cross_lang_all_features(dedirpath, "de", "class", itdirpath, "it")')
    do_cross_lang_all_features(dedirpath, "de", "class", itdirpath, "it")
    logging.info('Finished cross language from DE to IT do_cross_lang_all_features(dedirpath, "de", "class", itdirpath, "it")')
    logging.info('Starting cross language from DE to CZ do_cross_lang_all_features(dedirpath, "de", "class", czdirpath, "cz")')
    do_cross_lang_all_features(dedirpath, "de", "class", czdirpath, "cz")
    logging.info('Finished cross language from DE to CZ do_cross_lang_all_features(dedirpath, "de", "class", czdirpath, "cz")')
    logging.info("Finished cross language with DE as base language")

    logging.info("Starting cross language with IT as base language")
    logging.info('Starting cross language from IT to DE do_cross_lang_all_features(itdirpath, "it", "class", dedirpath, "de")')
    do_cross_lang_all_features(itdirpath, "it", "class", dedirpath, "de")
    logging.info('Finished cross language from IT to DE do_cross_lang_all_features(itdirpath, "it", "class", dedirpath, "de")')
    logging.info('Starting cross language from IT to CZ do_cross_lang_all_features(itdirpath, "it", "class", czdirpath, "cz")')
    do_cross_lang_all_features(itdirpath, "it", "class", czdirpath, "cz")
    logging.info('Finished cross language from IT to CZ do_cross_lang_all_features(itdirpath, "it", "class", czdirpath, "cz")')
    logging.info("Finished cross language with IT as base language")

    logging.info("Starting cross language with CZ as base language")
    logging.info('Starting cross language from CZ to DE do_cross_lang_all_features(czdirpath, "cz", "class", dedirpath, "de")')
    do_cross_lang_all_features(czdirpath, "cz", "class", dedirpath, "de")
    logging.info('Finished cross language from CZ to DE do_cross_lang_all_features(czdirpath, "cz", "class", dedirpath, "de")')
    logging.info('Starting cross language from CZ to IT do_cross_lang_all_features(czdirpath, "cz", "class", itdirpath, "it")')
    do_cross_lang_all_features(czdirpath, "cz", "class", itdirpath, "it")
    logging.info('Finished cross language from CZ to IT do_cross_lang_all_features(czdirpath, "cz", "class", itdirpath, "it")')
    logging.info("Finished cross language with CZ as base language")

    logging.info('Script End')


if __name__ == "__main__":
    main()

"""
TODO: Refactoring, reducing redundancy

"""

# print(getLexFeatures("/Users/sowmya/Research/CrossLing-Scoring/CrossLingualScoring/Datasets/DE-Parsed/1031_0003076_DE_C1.txt.parsed.txt", "de"))
# exit(1):
