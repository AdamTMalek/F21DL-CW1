from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from dataframe_manipulation import separate_x_and_y
import pandas as pd
from classes import classes


def print_report(y_test, y_pred):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("Multinomial Naive Bayes model accuracy(in %):",
          metrics.accuracy_score(y_test, y_pred)*100)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(data=conf_matrix, index=classes, columns=classes))
    classification_rprt = metrics.classification_report(
        y_test, y_pred, target_names=list(classes))
    print(classification_rprt)


def naive_bayes(images):
    x_data, y_data = separate_x_and_y(images)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data.values.ravel(), test_size=0.4, random_state=1)

    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(x_train, y_train)
    y_pred = naive_bayes_model.predict(x_test)
    print_report(y_test, y_pred)
