#!/usr/bin/python3
# -*- coding:utf-8 -*-
'''
@author: Eugenio Martínez Cámara
@since 03/05/2017
'''

import sys
import numpy as np
from nltk.tokenize.casual import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.preprocessing import sequence
from sklearn.svm.classes import LinearSVC
from keras.layers.embeddings import Embedding

TWEET_TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
CLASSES = []

def read_corpus(path):
    '''Load the corpus into memory
    '''
    
    ids = []
    labels = []
    tweets = []
    ids_append = ids.append
    classes_append = CLASSES.append
    labels_append = labels.append
    tweets_append = tweets.append
    with(open(path, 'r', encoding='utf-8')) as input_file:
        own_split = str.split
        own_strip = str.strip
        input_file.readline()
        for buffer in input_file:
            buffer_fields = own_split(buffer, ';;;')
            ids_append(own_strip(buffer_fields[0]))
            label = own_strip(buffer_fields[4])
            if(label not in CLASSES):
                classes_append(label)
            labels_append(CLASSES.index(label))
            tweets_append(own_strip(buffer_fields[-1]))
    
    return(ids, labels, tweets)

def tokenize(text):
    """Tokenize an input text
    
    Args:
        text: A String with the text to tokenize
    
    Returns:
        A list of Strings (tokens)
    """
    text_tokenized = TWEET_TOKENIZER.tokenize(text)
    return text_tokenized

def fit_transform_vocabulary(corpus):
    """Creates the vocabulary of the corpus
    
    Args:
        corpus: A list os str (documents)
        
    Returns:
        A tuple whose first element is a dictionary word-index and the second
        element is a list of list in which each position is the index of the 
        token in the vocabulary
    """
    
    vocabulary = {}
    corpus_indexes = []
    index = 1
    for doc in corpus:
        doc_indexes = []
        tokens = tokenize(doc)
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = index
                doc_indexes.append(index)
                index += 1
            else:
                doc_indexes.append(vocabulary[token])
        
            
        corpus_indexes.append(doc_indexes)
    return (vocabulary, corpus_indexes)


def classification_linear_svm(tweets, train_index, test_index, labels_train, random_state=None):
    """Classifies using SVM as classifier
    """
    
    
    #Representation
    tfidf_parser = TfidfVectorizer(tokenizer=tokenize, lowercase=False, analyzer='word')
    tweets_train = [tweets[tweet_index] for tweet_index in train_index]
    tweets_test = [tweets[tweet_index] for tweet_index in test_index]
    
    train_sparse_matrix_features_tfidf = tfidf_parser.fit_transform(tweets_train)
    test_sparse_matrix_features_tfidf = tfidf_parser.transform(tweets_test)
    
    
    classifier = LinearSVC(multi_class="ovr", random_state=random_state)
    print("Start SVM training")
    classifier = classifier.fit(train_sparse_matrix_features_tfidf, labels_train)
    print("Finish SVM training")
    y_labels = classifier.predict(test_sparse_matrix_features_tfidf)
    
    return y_labels
    


def classification_tfidf_rnn(tweets, train_index, test_index, labels_train, random_state=None):
    """Classification using a RNN with tfidf as features
    """
    #Representation
    tfidf_parser = TfidfVectorizer(tokenizer=tokenize, lowercase=False, analyzer='word')
    tweets_train = [tweets[tweet_index] for tweet_index in train_index]
    tweets_test = [tweets[tweet_index] for tweet_index in test_index]
    
    train_sparse_matrix_features_tfidf = tfidf_parser.fit_transform(tweets_train)
    test_sparse_matrix_features_tfidf = tfidf_parser.transform(tweets_test)
    
    train_features_tfidf = []
    own_train_features_tfidf_append = train_features_tfidf.append
    lengths_tweets = []
    own_lengths_tweets_append = lengths_tweets.append
    
    for tweet in train_sparse_matrix_features_tfidf:
        own_train_features_tfidf_append(tweet.data)
        own_lengths_tweets_append(len(tweet.data))
    

    test_features_tfidf = [tweet.data for tweet in test_sparse_matrix_features_tfidf]
    #Average length
    max_len_input = int(np.average(lengths_tweets, 0))
    #NN model
    nn_model = Sequential()
    nn_model.add(LSTM(32, input_shape=(max_len_input,1)))
    nn_model.add(Dense(len(CLASSES), activation='softmax'))
    nn_model.compile(optimizer="adam", 
                     loss="sparse_categorical_crossentropy", 
                     metrics=["accuracy"])
    
    train_features_tfidf_pad = sequence.pad_sequences(train_features_tfidf, maxlen=max_len_input, padding="post", truncating="post", dtype=type(train_features_tfidf[0][0]))
    train_features_tfidf_pad = np.expand_dims(train_features_tfidf_pad, axis=-1)
    print("Start RNN LSTM training")
    nn_model.fit(train_features_tfidf_pad, labels_train, batch_size=32, epochs=15, verbose=1)
    print("Finish RNN LSTM training")
    test_features_tfidf_pad = sequence.pad_sequences(test_features_tfidf, maxlen=max_len_input, padding="post", truncating="post", dtype=type(test_features_tfidf[0][0]))
    test_features_tfidf_pad = np.expand_dims(test_features_tfidf_pad, axis=-1)
    y_labels = nn_model.predict_classes(test_features_tfidf_pad, batch_size=32, verbose=1)
    
    return y_labels


def classifcation_embedings_rnn(corpus, train_index, test_index, labels_train, random_state=None):
    """Classification with RNN and embedings (no pre-trained)
    """
    
    #Build vocabulary and corpus indexes
    vocabulary, tweets_index = fit_transform_vocabulary(corpus)
    
    corpus_train = []
    own_corpus_train_append = corpus_train.append
    lengths_tweets = []
    own_lengths_tweets_append = lengths_tweets.append
    p = []
    for t_index in train_index:
        own_corpus_train_append(tweets_index[t_index])
        own_lengths_tweets_append(len(tweets_index[t_index]))
        p.append(corpus[t_index])
    max_len_input = int(np.average(lengths_tweets, 0))
    corpus_test = [tweets_index[t_index] for t_index in test_index]
    
    nn_model = Sequential()
    nn_model.add(Embedding(len(vocabulary)+1, 32, input_length=max_len_input))
    nn_model.add(LSTM(32))
    nn_model.add(Dense(len(CLASSES), activation='softmax'))
    nn_model.compile(optimizer="adam", 
                     loss="sparse_categorical_crossentropy", 
                     metrics=["accuracy"])
    
    
    train_features_tfidf_pad = sequence.pad_sequences(corpus_train, maxlen=max_len_input, padding="post", truncating="post", dtype=type(corpus_train[0][0]))
    
    
    print("Start RNN EMBEDDING LSTM training")
    nn_model.fit(train_features_tfidf_pad, labels_train, batch_size=32, epochs=15, verbose=1)
    print("Finish RNN EMBEDDING LSTM training")
    test_features_tfidf_pad = sequence.pad_sequences(corpus_test, maxlen=max_len_input, padding="post", truncating="post", dtype=type(corpus_train[0][0]))
    y_labels = nn_model.predict_classes(test_features_tfidf_pad, batch_size=32, verbose=1)
    return y_labels
                                                      
def calculate_quality_performamnce(y_labels, y_classified_labels, model_name):
    
    classes_index = [CLASSES.index(c) for c in CLASSES]
    accruacy = metrics.accuracy_score(y_labels, y_classified_labels)
    macro_precision = metrics.precision_score(y_labels, y_classified_labels, labels=classes_index, average="macro")
    macro_recall = metrics.recall_score(y_labels, y_classified_labels, labels=classes_index, average="macro")
    macro_f1 = metrics.f1_score(y_labels, y_classified_labels, labels=classes_index, average="macro")
    
    print("\n*** Results " + model_name + "***")
    print("Macro-Precision: " + str(macro_precision))
    print("Macro-Recall: " + str(macro_recall))
    print("Macro-F1: " + str(macro_f1))
    print("Accuracy: " + str(accruacy))

if __name__ == "__main__":
    
    
    np.random.seed(seed=7)
    
    input_file_path = sys.argv[1]
    
    #READ CORPUS
    ids, labels, tweets = read_corpus(input_file_path)
    
    
    
    #Representation
    #tfidf_parser = TfidfVectorizer(tokenizer=tokenize, lowercase=False, analyzer='word')
    #corpus_sparse_matrix_features_tfidf = tfidf_parser.fit_transform(tweets)
    
    
    train_index, test_index = train_test_split(np.arange(len(tweets)), test_size=0.2, random_state=7)
    labels_train = [labels[tweet_index] for tweet_index in train_index]
    labels_test = [labels[tweet_index] for tweet_index in test_index]
    
    
    y_labels_svn = classification_linear_svm(tweets, train_index, test_index, labels_train, 7)
    y_labels_rnn = classification_tfidf_rnn(tweets, train_index, test_index, labels_train)
    y_labels_embeddings_rnn = classifcation_embedings_rnn(tweets, train_index, test_index, labels_train) 
    calculate_quality_performamnce(labels_test, y_labels_svn, "SVM")
    calculate_quality_performamnce(labels_test, y_labels_rnn, "RNN_LSTM")
    calculate_quality_performamnce(labels_test, y_labels_embeddings_rnn, "RNN_EMBEDINGS_LSTM")
    
    