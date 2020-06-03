import bayes
import bayesBest
import sys

bayes = bayes.Bayes_Classifier()
best_bayes = bayesBest.Best_Bayes_Classifier()


def main():

    precision, recall, f_measure = bayes.classifier_performance(
        sys.argv[1])
    best_precision, best_recall, best_f_measure = best_bayes.classifier_performance(
        sys.argv[1])
    print("naive bayes:")
    print("precision: % s, recall: % s, f_measure: %s" %
          (precision, recall, f_measure))

    print("best bayes:")
    print("precision: % s, recall: % s, f_measure: %s" %
          (best_precision, best_recall, best_f_measure))


main()
