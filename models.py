# models.py

from sentiment_data import *
from utils import *

from collections import Counter
import numpy as np
import random
import string
import math

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self): 
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feat_vector = Counter()
        for word in sentence:
            word = word.lower()
            idx = -1 
            if word in STOPWORDS: 
                continue
            if all(char in string.punctuation for char in word): 
                continue 
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else: 
                idx = self.indexer.index_of(word)
                if idx == -1:
                    continue
            feat_vector[idx] = 1 
        return feat_vector

            
class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer 

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feat_vector = Counter()
        for i in range(1, len(sentence)):
            word = sentence[i].lower() 
            prev_word = sentence[i - 1].lower() 
            idx = -1 
            if word in STOPWORDS or prev_word in STOPWORDS: 
                continue
            if all(char in string.punctuation for char in word) or all(char in string.punctuation for char in prev_word): 
                continue
            if word[0] =='\'' or prev_word[0] == '\'': 
                continue
            bigram = prev_word + "|" + word
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
            else: 
                idx = self.indexer.index_of(bigram)
                if idx == -1:
                    continue
            feat_vector[idx] = 1 
        return feat_vector


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer, idfs):
        self.indexer = indexer
        self.idfs = idfs

    def get_indexer(self): 
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        tfs = {w : count / len(sentence) for w, count in Counter(sentence).items()}
        feat_vector = Counter()
        for word in sentence:
            idx = -1 
            if (tfs[word] * self.idfs.get(word, 1) < 0.1):
                continue
            if all(char in string.punctuation for char in word): 
                continue 
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else: 
                idx = self.indexer.index_of(word)
                if idx == -1:
                    continue
            feat_vector[idx] = 1 
        return feat_vector


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feat_vector = self.feat_extractor.extract_features(sentence)
        w_dot_feat = dot_product(self.weights, feat_vector)
        prediction = 1 if w_dot_feat > 0 else 0        
        return prediction
         

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feat_vector = self.feat_extractor.extract_features(sentence)
        w_dot_feat = dot_product(self.weights, feat_vector)
        prediction = 1 if w_dot_feat > 0 else 0        
        return prediction


def dot_product(weights, feat_vector): 
    return sum(weights[idx] * value for idx, value in feat_vector.items())

def update_weights_perceptron(weights, feat_vector, sign, alpha): 
    for idx in feat_vector: 
        weights[idx] = (weights[idx]  + sign * alpha * feat_vector[idx]) 

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    num_epochs = 80
    weights = np.zeros(100000)
    alpha = 0.01

    random.seed(26)
    random.shuffle(train_exs)
    for epoch in range(1, num_epochs + 1):
        if epoch % 20 == 0:
            print("------ This is epoch: " + str(epoch) + "------")
        for ex in train_exs: 
            feat_vector = feat_extractor.extract_features(ex.words, True)
            w_dot_feat = dot_product(weights, feat_vector)
            prediction = 1 if w_dot_feat > 0 else 0 
            if prediction != ex.label:
                if ex.label == 1:
                    update_weights_perceptron(weights, feat_vector, 1, alpha)
                else: 
                    update_weights_perceptron(weights, feat_vector, -1, alpha)
    return PerceptronClassifier(weights, feat_extractor)

            
def update_weights_logistic_regression(weights, feat_vector, alpha, label):
    w_dot_feat = dot_product(weights, feat_vector)
    prob_positive = 1/ (1 + math.exp(-1 * w_dot_feat))
    for idx in feat_vector: 
        weights[idx] = (weights[idx]  + alpha * (label - prob_positive) * feat_vector[idx])

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    num_epochs = 80
    weights = np.zeros(100000)
    alpha = 0.01

    random.seed(85)
    random.shuffle(train_exs)
    for epoch in range(1, num_epochs + 1):
        if epoch % 20 == 0:
            print("------ This is epoch: " + str(epoch) + "------")
        for ex in train_exs: 
            feat_vector = feat_extractor.extract_features(ex.words, True)
            w_dot_feat = dot_product(weights, feat_vector)
            update_weights_logistic_regression(weights, feat_vector, alpha, ex.label)
    return LogisticRegressionClassifier(weights, feat_extractor)

def get_idf_all_words(train_exs): 
    counts = Counter()
    N = len(train_exs)
    for ex in train_exs:
        for word in set(ex.words):
            counts[word] = counts.get(word, 0) + 1
    idfs = {w : math.log(N / count) for w, count in counts.items()}
    return idfs

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer(), get_idf_all_words(train_exs))
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model