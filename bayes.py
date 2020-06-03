import math
import os
import pickle
import re
import sys


class Bayes_Classifier:
    neg_dict = {}
    pos_dict = {}
    prior_p = {'neg_dict': 0, 'pos_dict': 0}

    def __init__(self, trainDirectory="movie_reviews/"):
        '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text.'''
        self.trainDirectory = trainDirectory

        if os.path.exists("pos_dict") and os.path.exists("neg_dict") and os.path.exists("prior_p"):
            self.pos_dict = self.load("pos_dict")
            self.neg_dict = self.load("neg_dict")
            self.prior_p = self.load("prior_p")
        else:
            self.train()

    def train(self):
        '''Trains the Naive Bayes Sentiment Classifier.'''
        lFileList = []
        neg_dict = self.neg_dict
        pos_dict = self.pos_dict
        prior_p = self. prior_p
        neg_review_sum = 0
        pos_review_sum = 0

        for fFileObj in os.walk("movies_reviews/"):
            lFileList = fFileObj[2]
            break

        for fileName in lFileList:
            if "-1-" in fileName:
                str_text = self.loadFile("movies_reviews/% s" % fileName)
                features = self.tokenize(str_text)

                for word in features:
                    if word in dict.keys(neg_dict):
                        neg_dict[word] += 1
                    else:
                        neg_dict[word] = 1

                neg_review_sum += 1

            if "-5-" in fileName:
                str_text = self.loadFile("movies_reviews/% s" % fileName)
                features = self.tokenize(str_text)

                for word in features:
                    if word in dict.keys(pos_dict):
                        pos_dict[word] += 1
                    else:
                        pos_dict[word] = 1

                pos_review_sum += 1

        prior_p['neg_dict'] = neg_review_sum / len(lFileList)
        prior_p['pos_dict'] = pos_review_sum / len(lFileList)

        self.save(neg_dict, "neg_dict")
        self.save(pos_dict, "pos_dict")
        self.save(prior_p, "prior_p")

        return neg_dict,

    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''
        neg_dict = self.neg_dict
        pos_dict = self.pos_dict
        prior_p = self.prior_p
        classification = None
        neg_log_sum = 0
        pos_log_sum = 0

        # total sum of frequency in each dictionary
        neg_f_sum = sum(neg_dict.values())
        pos_f_sum = sum(pos_dict.values())

        tokenized_text = self.tokenize(sText)
        # calculate target document probability
        for word in tokenized_text:
            neg_smooth_sum = len(tokenized_text) + neg_f_sum
            pos_smooth_sum = len(tokenized_text) + pos_f_sum

            # probability of each word in either dictionary
            neg_word_p = 0
            pos_word_p = 0

            # add-one smoothing
            if word in neg_dict.keys():
                # add-one smoothing
                neg_word_p = (neg_dict[word] + 1) / neg_smooth_sum
            else:
                neg_word_p = 1 / neg_smooth_sum

            # add-one smoothing
            if word in pos_dict.keys():
                # add-one smoothing
                pos_word_p = (pos_dict[word] + 1) / pos_smooth_sum
            else:
                pos_word_p = 1 / pos_smooth_sum

            # solving underflow issue
            neg_log_sum += math.log(neg_word_p)
            pos_log_sum += math.log(pos_word_p)

         # add log sum prior probability for each dictionary

        neg_log_sum += math.log(prior_p['neg_dict'])
        pos_log_sum += math.log(prior_p['pos_dict'])

        if round(neg_log_sum) > round(pos_log_sum):
            classification = "negative"
        elif round(neg_log_sum) < round(pos_log_sum):
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
        '''Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order).'''

        lTokens = []
        sToken = ""
        for c in sText:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))

        if sToken != "":
            lTokens.append(sToken)

        return lTokens

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

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f_measure = (2 * precision * recall) / (precision + recall)

        return precision, recall, f_measure


# bayes = Bayes_Classifier()
# result = bayes.classify("this movie is not good")
# print(result)

if __name__ == "__main__":
    bayes = Bayes_Classifier()
    bayes.classifier_performance(sys.argv[1])
