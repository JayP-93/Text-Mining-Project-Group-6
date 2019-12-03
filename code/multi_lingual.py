#!/usr/bin/python3

from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input, concatenate
from keras.layers import Embedding, Convolution1D, MaxPooling1D
from keras.utils import np_utils
import sys, time, re, glob
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from keras.layers import SpatialDropout1D

np.random.seed(1337)  # for reproducibility
seed = 1234
_white_spaces = re.compile(r"\s\s+")
maxlen = 2000
embedding_dims = 16
batch_size = 128
nb_epoch = 8
nb_filter = 50
filter_length = 5
pool_length = 5
minwordfreq = 15
mincharfreq = 0
maxwordlen = 400

# Any word character or not white space
tokenizer_re = re.compile("\w+|\S")

# Use this as arg
data_path = "..\Datasets"


def read_data():
    """
    Reads the data from the argv and transform it into a list of lists containing the documents, labels and lg_labels
    :return: a list with three lists: documents - list of the text from the documents,
                                      labels - list of the CEFR levels assigned,
                                      lg_labels - list of the language labels for the documents
    """
    labels = []
    lg_labels = []
    documents = []
    for data_file in glob.iglob(sys.argv[1] + "/*/*"):
        lang_label = data_file.split("/")[-2]

        if "RemovedFiles" in data_file: continue
        if "parsed" in data_file: continue
        doc = open(data_file, "r", encoding="utf-8").read().strip()
        wrds = doc.split(" ")

        label = data_file.split("/")[-1].split(".txt")[0].split("_")[-1]
        if label in ["EMPTY", "unrated"]: continue
        if len(wrds) >= maxwordlen:
            doc = " ".join(wrds[:maxwordlen])
        doc = _white_spaces.sub(" ", doc)
        labels.append(label)
        lg_labels.append(lang_label)
        documents.append(doc)

    return documents, labels, lg_labels


def char_tokenizer(s):
    """
    Transforms a string into a list of characters
    :param s: The string to be transformed
    :return: List of the characters of the string
    """
    return list(s)


def word_tokenizer(s):
    """
    Transforms a string into a list of words
    :param s: The string to be transformed
    :return: List of the words of the string
    """
    return tokenizer_re.findall(s)


def getWords(D):
    """
    :param D: a list of documents (strings)
    :return: wordSet: a dictionary with the frequence of the words in the list;
             max_features: 3 + number of words that have a frequence > to minwordfreq
    """
    wordSet = defaultdict(int)
    max_features = 3
    for d in D:
        for c in word_tokenizer(d):
            wordSet[c] += 1
    for c in wordSet:
        if wordSet[c] > minwordfreq:
            max_features += 1
    return wordSet, max_features


def getChars(D):
    """
    :param D: a list of documents (strings)
    :return: charSet: a dictionary with the frequence of the characters in the list;
             max_features: 3 + number of characters that have a frequence > to mincharfreq
    """
    charSet = defaultdict(int)
    max_features = 3
    for d in D:
        for c in char_tokenizer(d):
            charSet[c] += 1
    for c in charSet:
        if charSet[c] > mincharfreq:
            max_features += 1
    return charSet, max_features


def transform(D, vocab, minfreq, tokenizer="char"):
    """
    Transforms the documents in a list of lists of integers where every list corresponds to a document
    :param D: list of documents
    :param vocab: the vocabulary to be used
    :param minfreq: the minimum frequence for a item of the vocabulary to be considered feature
    :param tokenizer: the token to be considered for a document
    :return: X list of lists of integers where every list is a document and :
                                        1 indicates the beginning of a document
                                        2 indicates a non-feature item
                                        > 3 indicates a feature with the given index
    """
    features = defaultdict(int)
    count = 0
    # Gives an index to every features (vocab element with frequence > minfreq)
    # The indexes begin at 0 increase by 1 for every feature found
    for i, k in enumerate(vocab.keys()):
        if vocab[k] > minfreq:
            features[k] = count
            count += 1

    start_char = 1
    oov_char = 2
    index_from = 3
    # X is the list of lists of integers where every list is a document and
    # start_char indicates the beginning of a document
    # oov_char represent a vocab element that is not a feature
    # value different from oov_char and start_char - it is the index of a feature+3
    X = []
    for j, d in enumerate(D):
        # x is the list of integers where if the value is oov_char then it is not a feature
        # if the value is start_char - it indicates the beginning of the document
        # if the value is different from oov_char and start_char - it is the index of a feature+3
        x = [start_char]
        # z is the list of vocab elements for every document
        z = None
        if tokenizer == "word":
            z = word_tokenizer(d)
        else:
            z = char_tokenizer(d)
        for c in z:
            freq = vocab[c]
            if c in vocab:
                if c in features:
                    x.append(features[c] + index_from)
                else:
                    x.append(oov_char)
            else:
                continue
        X.append(x)
    return X


print("Reading the training set... ", end="")
sys.stdout.flush()
pt = time.time()
doc_train, y_labels, y_lang_labels = read_data()
print(time.time() - pt)

print("Transforming the datasets... ", end="")
sys.stdout.flush()
pt = time.time()
word_vocab, max_word_features = getWords(doc_train)
char_vocab, max_char_features = getChars(doc_train)
print("Number of word features= ", max_word_features, " char features= ", max_char_features)
x_char_train = transform(doc_train, char_vocab, mincharfreq, tokenizer="char")
x_word_train = transform(doc_train, word_vocab, minwordfreq, tokenizer="word")
print(len(x_char_train), 'train sequences')
print(time.time() - pt)

print('Pad sequences (samples x time)')
# Transforms the lists in 2D array where every row is a list of the list (a document)
# If the list is smaller than maxlen, it adds 0 elements at the beginning till it gets to maxlen
x_char_train = sequence.pad_sequences(x_char_train, maxlen=maxlen)
x_word_train = sequence.pad_sequences(x_word_train, maxlen=maxwordlen)
print('x_train shape:', x_char_train.shape)

print("Transforming the labels... ", end="")
sys.stdout.flush()
pt = time.time()
unique_labels = list(set(y_labels))
lang_labels = ["CZ", "IT", "DE", "EN"]
print("Class labels = ", unique_labels)
n_classes = len(unique_labels)
n_grp_classes = len(lang_labels)

grp_train, grp_test = [], []

# Transforms y_labels and y_lang_labels to lists of integer where the integer corresponds
# to the index of the label unique_labels or lang_labels
y_labels = [unique_labels.index(y) for y in y_labels]
y_lang_labels = [lang_labels.index(y) for y in y_lang_labels]

# One-hot-encoding for every value of the list y_labels, y_lang_labels
y_train = np_utils.to_categorical(np.array(y_labels), len(unique_labels))
lng_train = np_utils.to_categorical(np.array(y_lang_labels), len(lang_labels))

print(time.time() - pt)

cv_accs, cv_f1 = [], []
# Cross-validation object. Provides train/test indices to split the data
k_fold = StratifiedKFold(10, random_state=seed)
n_iter = 1
all_golds = []
all_preds = []

for train, test in k_fold.split(x_word_train, y_labels):
    print('Build model... ', n_iter)
    char_input = Input(shape=(maxlen,), dtype='int32', name='char_input')
    charx = Embedding(max_char_features, 16, input_length=maxlen)(char_input)
    charx = SpatialDropout1D(0.25)(charx)

    word_input = Input(shape=(maxwordlen,), dtype='int32', name='word_input')
    wordx = Embedding(max_word_features, 32, input_length=maxwordlen)(word_input)
    wordx = SpatialDropout1D(0.25)(wordx)

    charx = Flatten()(charx)
    wordx = Flatten()(wordx)
    y = concatenate([charx, wordx])
    grp_predictions = Dense(n_grp_classes, activation='softmax')(y)
    y = concatenate([y, grp_predictions])

    y = Dropout(0.25)(y)
    y_predictions = Dense(n_classes, activation='softmax')(y)

    model = Model(inputs=[char_input, word_input], outputs=[y_predictions, grp_predictions])

    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'], loss_weights=[1.0, 0.5])

    hist = model.fit([x_char_train[train], x_word_train[train]], [y_train[train], lng_train[train]],
                     batch_size=batch_size,
                     epochs=nb_epoch)

    y_pred = model.predict([x_char_train[test], x_word_train[test]])
    print(y_pred[0].shape, y_pred[1].shape, sep="\n")
    y_classes = np.argmax(y_pred[0], axis=1)
    y_gold = np.array(y_labels)[test]
    print(y_classes.shape, y_gold.shape)

    pred_labels = [unique_labels[x] for x in y_classes]
    gold_labels = [unique_labels[x] for x in y_gold]
    all_golds.extend(gold_labels)
    all_preds.extend(pred_labels)
    cv_f1.append(f1_score(y_gold, y_classes, average="weighted"))
    print(confusion_matrix(gold_labels, pred_labels, labels=unique_labels))
    n_iter += 1

print("\nF1-scores", cv_f1, sep="\n")
print("Average F1 scores", np.mean(cv_f1))
print(confusion_matrix(all_golds, all_preds))
