from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
from dataframe_manipulation import separate_x_and_y
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classes import classes
from dataframe_manipulation import load_image_data, shuffle_dataframe, load_image_data_with_ten_one


def coord_plot(x_data, y_pred, k):
    x_data.loc[:, 'Cluster'] = y_pred
    for x in range(0, k):
        only_class = x_data.loc[x_data['Cluster'] == x]
        plot_label = "Image data clustered as " + str(x)
        fig, ax = plt.subplots(1)
        fig = pd.pandas.plotting.parallel_coordinates(
            only_class.head(50), color='red', class_column='Cluster', linewidth=0.3, axvlines=False)

        # Plot formatting
        ax.set_xlabel('Pixel Position')
        ax.set_ylabel('Pixel Value')
        ax.set_xticklabels([])
        ax.set_title(plot_label)
        plt.show()


def parallel_coord_plot(x_data, y_pred):
    x_data.loc[:, 'Cluster'] = y_pred
    fig, ax = plt.subplots(1)

    fig = pd.pandas.plotting.parallel_coordinates(
        # x_data.iloc[:, 1550:1601], 'Cluster')
        x_data.head(25), 'Cluster', linewidth=0.3, axvlines=False)
    ax.set_xlabel('Pixel Position')
    ax.set_ylabel('Pixel Value')
    ax.set_xticklabels([])
    ax.set_title("Image Data Parallel Coordinate Plot")

    plt.show()


def find_best_k(x_data):
    kmeans_per_k = [KMeans(n_clusters=k, random_state=1).fit(x_data)
                    for k in range(1, 10)]
    inertias = [model.inertia_ for model in kmeans_per_k]
    index_of_largest = inertias.index(min(inertias))
    return index_of_largest


def k_means_clustering(images):
    x_data, _ = separate_x_and_y(images)  # Extracts x data from image
    k = find_best_k(x_data)
    kmeans = KMeans(n_clusters=k, random_state=1)
    y_pred = kmeans.fit_predict(x_data)
    print("The kmeans inertia (lower is better) is", kmeans.inertia_)
    parallel_coord_plot(x_data, y_pred)
    coord_plot(x_data, y_pred, k)


images = load_image_data(
    "data/x_train_gr_smpl_pruned_20.csv", "data/y_train_smpl.csv")  # Used for naive bayes
shuffled_images = shuffle_dataframe(images)


k_means_clustering(shuffled_images)
