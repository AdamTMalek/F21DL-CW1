from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


def naive_bayes(images):

    x_data = images.iloc[:, images.columns != '0']
    y_data = images.iloc[:, -1:]

    print(x_data)
    print(y_data)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data.values.ravel(), test_size=0.4, random_state=1)

    gnb = MultinomialNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    print("Gaussian Naive Bayes model accuracy(in %):",
          metrics.accuracy_score(y_test, y_pred)*100)
