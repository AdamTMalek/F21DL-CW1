from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from dataframe_manipulation import separate_x_and_y
import pandas as pd
from classes import classes
from image_cropper import crop_images
from dataframe_manipulation import load_image_data, shuffle_dataframe, load_image_data_with_ten_one
from get_top_correlating_pixels import get_top_corr_pixels, remove_none_corr_pixels

top_corr_pixels = [5, 10, 20]


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


def main():
    crop_images("data/x_train_gr_smpl.csv", 40)
    images = load_image_data(
        "data/x_train_gr_smpl_40.csv", "data/y_train_smpl.csv")  # Used for naive bayes
    corr_images = load_image_data_with_ten_one(
        "data/x_train_gr_smpl_40.csv")  # Used for finding top correlating pixels

    shuffled_images = shuffle_dataframe(images)
    shuffled_corr_images = shuffle_dataframe(corr_images)
    print("=========== Unpruned ================")
    naive_bayes(shuffled_images)
    for x in top_corr_pixels:
        print("=========== Top", x, "pixels ================")
        top_pixels = get_top_corr_pixels(shuffled_corr_images, x)
        pruned_images = remove_none_corr_pixels(shuffled_images, top_pixels)
        naive_bayes(pruned_images)


if __name__ == "__main__":
    main()