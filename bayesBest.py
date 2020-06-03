import math
import os
import pickle
import re
import sys


class Best_Bayes_Classifier:
    pos_model = {}
    neg_model = {}
    Vtot = []

    def __init__(self, trainDirectory="movies_reviews/"):
        '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text.'''
        self.trainDirectory = trainDirectory

        if os.path.exists("pos_model") and os.path.exists("neg_model"):
            self.pos_model = self.load("pos_model")
            self.neg_model = self.load("neg_model")
            if not os.path.exists("Vtot"):
                self.getVtot()

            else:
                self.Vtot = self.load("Vtot")
        else:
            self.train()

    def getVtot(self):
        corpus_pos_train = ""
        corpus_neg_train = ""
        N_pos = 0
        N_neg = 0
        N_doc = 0

        for filename in os.listdir('movies_reviews'):
            filepath = os.path.join('movies_reviews', filename)
            if os.path.isfile(filepath):
                N_doc += 1
                # if positive, add to positive
                if "-5-" in filename:
                    text = ' '.join(open(filepath, 'r').readlines())
                    corpus_pos_train += '<S> '+text+' '  # add "<S>" to indicate start of article
                    N_pos += 1
                # if negative, add to negative
                elif "-1-" in filename:
                    text = ' '.join(open(filepath, 'r').readlines())
                    corpus_neg_train += '<S> '+text+' '  # add "<S>" to indicate start of article
                    N_neg += 1

        # pos_unigrams = self.create_ngrams(1, self.tokenize(corpus_pos_train))
        # neg_unigrams = self.create_ngrams(1, self.tokenize(corpus_neg_train))
        pos_bigrams = self.create_bigrams(2, self.tokenize(corpus_pos_train))
        neg_bigrams = self.create_bigrams(2, self.tokenize(corpus_neg_train))

        # whole vocabulary (unique tokens)
        # Vtot = list(set(pos_bigrams + neg_bigrams))
        Vtot = [list(x) for x in set(tuple(x)
                                     for x in (pos_bigrams+neg_bigrams))]

        self.save(Vtot, "Vtot")

    def train(self):
        corpus_pos_train = ""
        corpus_neg_train = ""
        N_pos = 0
        N_neg = 0
        N_doc = 0

        for filename in os.listdir('movies_reviews'):
            filepath = os.path.join('movies_reviews', filename)
            if os.path.isfile(filepath):
                N_doc += 1
                # if positive, add to positive
                if "-5-" in filename:
                    text = ' '.join(open(filepath, 'r').readlines())
                    corpus_pos_train += '<S> '+text+' '  # add "<S>" to indicate start of article
                    N_pos += 1
                # if negative, add to negative
                elif "-1-" in filename:
                    text = ' '.join(open(filepath, 'r').readlines())
                    corpus_neg_train += '<S> '+text+' '  # add "<S>" to indicate start of article
                    N_neg += 1

        pos_unigrams = self.create_ngrams(1, self.tokenize(corpus_pos_train))
        neg_unigrams = self.create_ngrams(1, self.tokenize(corpus_neg_train))
        pos_bigrams = self.create_bigrams(2, self.tokenize(corpus_pos_train))
        neg_bigrams = self.create_bigrams(2, self.tokenize(corpus_neg_train))

        # whole vocabulary (unique tokens)
        # Vtot = list(set(pos_bigrams + neg_bigrams))
        Vtot = [list(x) for x in set(tuple(x)
                                     for x in (pos_bigrams+neg_bigrams))]
        # smoothing factor
        k = 1

        # train
        pos_model = self.train_model(
            pos_bigrams, pos_unigrams, Vtot, N_pos, N_doc, k)
        neg_model = self.train_model(
            neg_bigrams, neg_unigrams, Vtot, N_neg, N_doc, k)

        self.save(pos_model, "pos_model")
        self.save(neg_model, "neg_model")
        self.save(Vtot, "Vtot")
        # print(neg_model)
        return pos_model, neg_model, Vtot

    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''

        pos_model = self.pos_model
        neg_model = self.neg_model
        Vtot = self.Vtot

        prob_pos_model = pos_model[0]
        prob_neg_model = neg_model[0]

        prob_pos = 0
        prob_neg = 0

        test_token = self.create_bigrams(2, self.tokenize(sText))

        # calculate target document probability
        for bigram in test_token:
            word = bigram[1]
            wprev = bigram[0]

            if bigram in Vtot:  # ignore if not in trained vocab
                # add loglikelihood for word from trained model
                prob_pos_model += pos_model[1][word][wprev]
                # add loglikelihood for word from trained model
                prob_neg_model += neg_model[1][word][wprev]

        prob_pos = prob_pos_model
        prob_neg = prob_neg_model

        if prob_neg > prob_pos:
            classification = "negative"
        elif prob_neg < prob_pos:
            classification = "positive"
        else:
            classification = "neutral"

        return classification

    def loadFile(self, sFilename):
        '''Given a file name, return the contents of the file as a string.'''

        f = open(sFilename, "r")
        sTxt = f.read()
        f.close()
        return sTxt

    def save(self, dObj, sFilename):
        '''Given an object and a file name, write the object to the file using pickle.'''

        f = open(sFilename, "wb")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()

    def load(self, sFilename):
        '''Given a file name, load and return the object stored in the file.'''

        f = open(sFilename, "rb")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        # remove newline character
        sText = re.sub('\n', ' ', sText)
        # remove apostrophes (e.g. so 'haven't -> havent')
        sText = re.sub('\'', ' \'', sText)
        # replace all punctuation with spaces (leave apostrophes and <S> alone)
        words = re.sub(r'[^\w\s\'\<S>]', ' ', sText)
        normalized = words.lower()
        # tokenize
        tokens = normalized.split(' ')
        # remove empty-strings
        tokens = [tok for tok in tokens if tok != '']

        return tokens

    def train_model(self, bigrams, unigrams, Vtot, N_class, N_doc, k):
        # P(c) = logprior is log(docs in class c/tot docs) -> MLE!
        logprior = math.log((float(N_class)/float(N_doc)), 2)

        # V = Vtot -> total vocabulary disregarding classes
        # bigdoc[c] = unigrams -> all unigrams from docs in class c
        # smoothing factor k to avoid negative probabilities

        loglikelihood = {}  # this will be for each follow-word using bigram info
        for bigram in Vtot:  # all (unique) bigrams: (wi-1,wi)
            w_prev = bigram[0]
            word = bigram[1]
            # check if word already in loglikelihood: means that all possible bigrams ending with it have been added!
            # if yes, move on. Else add it!
            if not(word in loglikelihood):
                # for w: get all possible w_prev & likelihoods
                wprevs_for_w = [wprev for (wprev, w) in Vtot if (w == word)]

                w_prev_dict = {}
                for wprev in wprevs_for_w:

                    count_tuple = float(bigrams.count([wprev, word]))
                    count_wprev = float(unigrams.count(wprev))

                    # logLIKELIHOOD of wprev|word #
                    w_prev_dict[wprev] = math.log(
                        ((count_tuple+k)/(count_wprev+(k*len(Vtot)))), 2)

                loglikelihood[word] = w_prev_dict
                # look up any wprev|word with: loglikelihood[word][wprev]

        return logprior, loglikelihood

    def create_ngrams(self, n, tokens):
        ngram_list = []
        for tok_location, tok in enumerate(tokens):
            # stops at last possible index for ngram length
            if tok_location <= len(tokens)-n:
                ngram = ''
                for i in range(n):  # n=3 -> [0,1,2]
                    ngram += tokens[tok_location+i]
                    ngram += ' '
                ngram_list.append(ngram)
        return ngram_list

    def create_bigrams(self, n, tokens):  # for different bigram format (tuple not string)
        ngram_list = []
        for tok_location, tok in enumerate(tokens):
            # stops at last possible index for ngram length
            if tok_location <= len(tokens)-n:
                ngram = []
                for i in range(n):  # n=3 -> [0,1,2]
                    ngram.append(tokens[tok_location+i])
                ngram_list.append(ngram)
        return ngram_list

    def classifier_performance(self, tartgetDirectory):
        lFileList = []
        false_neg = 0
        true_pos = 0
        false_pos = 0

        for fFileObj in os.walk(tartgetDirectory):
            lFileList = fFileObj[2]
            break

        for fileName in lFileList:
            file_str = self.loadFile("% s/% s" % (tartgetDirectory, fileName))
            file_class = self.classify(file_str)

            # false-positive
            if "-1-" in fileName and file_class == "positive":
                false_pos += 1

            if "-5-" in fileName and file_class == "positive":
                true_pos += 1

            if "-5-" in fileName and file_class == "negative":
                false_neg += 1

        precision = true_pos / float(true_pos + false_pos)
        recall = true_pos / float(true_pos + false_neg)
        f_measure = (2 * precision * recall) / float(precision + recall)

        return precision, recall, f_measure


# bayes = Best_Bayes_Classifier()
# result = bayes.classify("I hate this movie")
# print(result)

if __name__ == "__main__":
    bayes = Best_Bayes_Classifier()
    bayes.classifier_performance(sys.argv[1])
