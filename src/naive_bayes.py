import pandas as pd
import numpy as np
from src.get_dataframes import load_image_data_without_y_data, load_all_y_data


def process_images():
    x_data = load_image_data_without_y_data()
    x_data = (x_data - x_data.mean()) / x_data.std()
    median_values = x_data.median(axis=1)
    x_data = x_data.transpose()
    for i in range(len(median_values)):
        # print(median_values[i])
        # print(x_data[i])
        x_data[i] = x_data[i].apply(lambda x: 0 if x <= median_values[i] else 1)
    x_data = x_data.transpose()
    x_data.to_csv("out.csv")
    print(x_data)
    # return x_data


def separate_images_to_csv():
    x_data = pd.read_csv("../data/x_train_gr_smpl.csv")
    x_data = x_data.transpose()
    for i in range(len(x_data.columns)):
        column = x_data[i].to_numpy()
        image = np.array_split(column, 48)
        df = pd.DataFrame(image)
        df.to_csv("../data/individual images/" + str(i) + ".csv")


def naive_bayes():
    # Binarise x data and flatten with y
    x_data = process_images()
    y_data = load_all_y_data()
    xy_concat = pd.concat([x_data.reset_index(drop=True), y_data.reset_index(drop=True)], axis=1)
    xy_concat = xy_concat.astype('int64')

    # Bayes
    print("Length of xy data: " + str(len(xy_concat)))
    pSign = xy_concat['Speed Limit 20'].value_counts() / len(xy_concat)
    print(pSign)

    p = pSign[(0)]
    print("P: " + str(p))
    print("x_data columns: " + str(len(x_data.columns)))
    # for i in range(len(x_data.columns)):
    #     pPixelGivenSign = xy_concat.groupby([str(i), "Speed Limit 20"]).size().div(len(xy_concat))[(1, 0)]
    #     # p *= pPixelGivenSign
    #     print(pPixelGivenSign)git
    # print(p)

