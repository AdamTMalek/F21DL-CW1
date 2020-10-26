from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from dataframe_manipulation import separate_x_and_y


def naive_bayes(images):
    x_data, y_data = separate_x_and_y(images)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data.values.ravel(), test_size=0.4, random_state=1)

    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(x_train, y_train)
    y_pred = naive_bayes_model.predict(x_test)
    print("Gaussian Naive Bayes model accuracy(in %):",
          metrics.accuracy_score(y_test, y_pred)*100)
