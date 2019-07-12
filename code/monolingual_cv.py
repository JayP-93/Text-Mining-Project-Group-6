from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input, merge
from keras.layers import Embedding, Convolution1D, MaxPooling1D
from keras.layers import AveragePooling1D, LSTM, GRU
from keras.utils import np_utils
import sys, time, re, glob
from collections import defaultdict
from gensim.utils import simple_preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

# Just a regex for finding all whitespace characters [ \t\n\r\f\v]
_white_spaces = re.compile(r"\s\s+")

maxchars = 200
embedding_dims = 100
batch_size = 32
nb_epoch = 10
nb_filter = 128
filter_length = 5
pool_length = 32
minfreq = 0
data_path = sys.argv[1]
minwordfreq = 15
maxwordlen = 400
np.random.seed(1337)  # for reproducibility
seed = 1234


# Use this as arg
# data_path = "..\Datasets\CZ"

def read_data():
    """
    Reads the data from the argv and transform it into a tuple of lists containing the documents, labels
    :return: documents - list of the processed text from the documents,
             labels - list of the CEFR levels assigned
    """
    labels = []
    documents = []
    for data_file in glob.iglob(sys.argv[1] + "/*"):
        doc = open(data_file, "r", encoding="utf8").read().strip()
        wrds = doc.split(" ")
        # extract label from filename
        label = data_file.split("/")[-1].split(".txt")[0].split("_")[-1]
        if label == "EMPTY":
            continue
        # limit the number of words in the text to maxwordlen
        if len(wrds) >= maxwordlen:
            doc = " ".join(wrds[:maxwordlen])
        # substitute whitespace characters with an empty character
        doc = _white_spaces.sub(" ", doc)
        labels.append(label)
        documents.append(doc)

    return documents, labels


def char_tokenizer(s):
    """
    Transforms a string into a list of characters
    :param s: The string to be transformed
    :return: List of the characters of the string
    """
    return list(s)


def word_tokenizer(s):
    """
    Documentation partially taken from gensim.utils.simple_preprocess
    Convert a document into a list of lowercase tokens, ignoring tokens that are too short (<2) or too long (>15)
    :param s: The string to be transformed
    :return: List of the words of the string
    """
    return simple_preprocess(s)


def getWords(D):
    """
    :param D: a list of documents (strings)
    :return: wordSet: a dictionary with the frequency of the words in the list;
             max_features: 3 + number of words that have a frequency > to minwordfreq
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
    :return: charSet: a dictionary with the frequency of the characters in the list;
             max_features: 3 + number of characters that have a frequency > to mincharfreq
    """
    charSet = defaultdict(int)
    max_features = 3
    for d in D:
        for c in char_tokenizer(d):
            charSet[c] += 1
    for c in charSet:
        if charSet[c] > minfreq:
            max_features += 1
    return charSet, max_features


def transform(D, vocab, minfreq, tokenizer="char"):
    """
    Transforms the documents in a list of lists of integers where every list corresponds to a document
    :param D: list of documents
    :param vocab: the vocabulary to be used
    :param minfreq: the minimum frequency for an item of the vocabulary to be considered feature
    :param tokenizer: the token (char or word) to be considered for a document
    :return: X list of lists of integers where every list is a document and :
                                        1 indicates the beginning of a document
                                        2 indicates a non-feature item
                                        > 3 indicates a feature with the given index
    """
    features = defaultdict(int)
    count = 0
    # Gives an index to every feature (vocab element with frequency > minfreq)
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
        # z is the list of vocab elements(words or characters depending on the tokenizer argument) for every document
        if tokenizer == "word":
            z = word_tokenizer(d)
        else:
            z = char_tokenizer(d)
        for c in z:
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
doc_train, y_labels = read_data()
print(time.time() - pt)

print("Transforming the datasets... ", end="")
sys.stdout.flush()
pt = time.time()

word_vocab, max_word_features = getWords(doc_train)
print("Number of features= ", max_word_features)
x_word_train = transform(doc_train, word_vocab, minwordfreq, tokenizer="word")

print(len(x_word_train), 'train sequences')

print(time.time() - pt)

print('Pad sequences (samples x time)')
# Transforms the lists in 2D array where every row is a list of integers (a document)
# If the list is smaller than maxwordlen, it adds 0 elements at the beginning till it gets to maxwordlen
x_word_train = sequence.pad_sequences(x_word_train, maxlen=maxwordlen)
print('x_train shape:', x_word_train.shape)

print("Transforming the labels... ", end="")
sys.stdout.flush()
pt = time.time()
unique_labels = list(set(y_labels))
print("Class labels = ", unique_labels)
n_classes = len(unique_labels)
indim = x_word_train.shape[1]

# Transforms y_labels to a list of integer where the integer corresponds to the index of the label in unique_labels
y_labels = [unique_labels.index(y) for y in y_labels]

# One-hot-encoding for every value of the list y_labels
y_train = np_utils.to_categorical(np.array(y_labels), len(unique_labels))

print('y_train shape:', y_train.shape)

print(time.time() - pt)

cv_f1 = []
# Cross-validation object. Provides train/test indices to split the data
k_fold = StratifiedKFold(10, random_state=seed)
all_gold = []
all_preds = []
for train, test in k_fold.split(x_word_train, y_labels):
    # print("TRAIN:", train, "TEST:", test)
    print('Build model...')

    # Setting up model
    model = Sequential()
    model.add(Embedding(max_word_features, embedding_dims, input_length=maxwordlen))
    # model.add(GRU(50))
    # model.add(AveragePooling1D(pool_length=8))
    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # Train model
    model.fit(x_word_train[train], y_train[train],
              batch_size=batch_size,
              epochs=nb_epoch)

    # Do some predictions
    y_pred = model.predict_classes(x_word_train[test])
    # print(y_pred, np.array(y_labels)[test], sep="\n")

    pred_labels = [unique_labels[x] for x in y_pred]
    gold_labels = [unique_labels[x] for x in np.array(y_labels)[test]]
    all_gold.extend(gold_labels)
    all_preds.extend(pred_labels)

    # Calculate F1 score for this fold and add it to the list of F1 scores for all folds
    cv_f1.append(f1_score(np.array(y_labels)[test], y_pred, average="weighted"))
    # Print confusion matrix
    print(confusion_matrix(gold_labels, pred_labels, labels=unique_labels))

print("\nF1-scores", cv_f1, sep="\n")
print("Average F1 scores", np.mean(cv_f1))
print(confusion_matrix(all_gold, all_preds))
