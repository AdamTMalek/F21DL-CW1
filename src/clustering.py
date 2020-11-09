import sys
import time
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
import random
import argparse
from nparray import ndarray
from pandas import DataFrame
from sklearn import metrics
from pathlib import Path
from dataframe_manipulation import load_image_data, shuffle_dataframe
from dataframe_manipulation import separate_x_and_y
from sklearn.cluster import KMeans
from tabulate import tabulate


def single_image_line_plot(x_data: DataFrame, y_pred: DataFrame, k: int, show: bool) -> None:
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
        plt.savefig(f'../evidence/K-Means/Line Screenshots/Cluster_{x}')
        if show:
            plt.show()


def multi_image_line_plot(x_data: DataFrame, y_pred: DataFrame, show: bool, normalised: bool) -> None:
    x_data.loc[:, 'Cluster'] = y_pred
    fig, ax = plt.subplots(1)

    fig = pd.pandas.plotting.parallel_coordinates(
        x_data.head(25), 'Cluster', linewidth=0.3, axvlines=False)
    ax.set_xlabel('Pixel Position')
    ax.set_ylabel('Pixel Value')
    ax.set_xticklabels([])
    ax.set_title("Image Data Parallel Coordinate Plot")
    if normalised:
        plt.savefig(f'../evidence/K-Means/Image Data Parallel Coordinate Plot (Normalised)')
    else:
        plt.savefig(f'../evidence/K-Means/Image Data Parallel Coordinate Plot')
    if show:
        plt.show()


def plot_entire_clusters(cluster: Dict, cluster_center: Dict, normalised: bool, show: bool) -> None:
    """"
    Plots all entries in cluster onto a scatter graph and saves it. Use show param to display the graph

    :param cluster: Dictionary where key is cluster id and value is the images clustered into the cluster
    :param cluster_center: Dictionary where key is cluster id and value is the pixel values of the center point of each
    cluster
    :param normalised: To determine if result needs to be saved to normalised folder
    :param show: To determine if the graph needs to be displayed
    """
    for i in range(len(cluster)):
        x_values = cluster[i].keys()
        pics = cluster[i].values
        fig = plt.figure(f'Cluster {i}')
        ax = fig.add_subplot(111)
        for val in pics:
            ax.scatter(x_values[:], val[:], zorder=1)
        ax.scatter(x_values[:], cluster_center[i][:], marker="x", zorder=2, label="Cluster centers")
        ax.set_xlabel('Pixel Position')
        ax.set_ylabel('Pixel Value')
        ax.set_title(f'Cluster {i}\'s images relative to cluster center')
        ax.legend(loc='upper right')
        if normalised:
            plt.savefig(f'../evidence/K-Means/Cluster Screenshots/Normalised/Cluster_{i}.png')
        else:
            plt.savefig(f'../evidence/K-Means/Cluster Screenshots/Not Normalised/Cluster_{i}.png')
        if show:
            plt.show()
        plt.close(fig)


def find_best_k(x_data: DataFrame) -> int:
    """"
    Finds best number of clusters

    :param x_data: Image data
    """
    kmeans_per_k = [KMeans(n_clusters=k).fit(x_data) for k in range(1, 13)]
    inertias = [model.inertia_ for model in kmeans_per_k]
    index_of_largest = inertias.index(min(inertias))
    return index_of_largest


def cluster_to_csv(cluster: DataFrame, cluster_int: int, normalise: bool) -> None:
    """
    Saves cluster data to csv (for viewing in image_viewer.py)

    :param cluster: Image data that has been clustered into the cluster
    :param cluster_int: Cluster id
    :param normalise: Boolean to determine where to save
    """
    if normalise:
        name = f'cluster_{cluster_int}_normalised'
    else:
        name = f'cluster_{cluster_int}'
    cluster.to_csv(f'../data/{name}.csv', compression=None, index=False)


def get_clustered_images(x_data: DataFrame, cluster_array: ndarray) -> Dict:
    """
    Splits images into their assigned clusters determine by k-means algorithm

    :param x_data: Image data
    :param cluster_array: Assigned clusters from k-means
    :return: Dictionary where key is cluster id and value is images that is grouped into the cluster
    """
    clusters = pd.Series(cluster_array)
    groups = pd.concat([x_data, clusters.rename('cluster')], axis=1)
    result = {}
    for i in range(max(cluster_array)+1):
        result[i] = groups.loc[groups['cluster'] == i]
        result[i] = result[i].drop(['cluster'], axis=1)
    return result


def accuracy_of_clusters(y_data: DataFrame, new_groups: ndarray) -> Dict:
    """
    Determines the accuracy of the cluster

    :param y_data: Original group assignments
    :param new_groups: k-means cluster assignments
    :return: Dictionary of cluster accuracy and the overall rand score
    """
    new = pd.Series(new_groups)
    y_data['new'] = new
    clusters = {}
    for i in range(max(y_data['new'])):
        clusters[i] = y_data.loc[y_data['new'] == i]
    accuracy = {}
    total_accuracy = 0
    for i in range(len(clusters.keys())):
        group = clusters[i]
        count = group.loc[(group['original'] == i) & (group['new'] == i)].shape[0]
        total_accuracy += count
        accuracy[i] = (count / len(group['original'])) * 100
    accuracy['rand'] = metrics.adjusted_rand_score(y_data['original'], y_data['new'])
    accuracy['total'] = (total_accuracy / y_data.shape[0]) * 100
    return accuracy


def k_means_clustering(images: DataFrame, normalise: bool) -> Dict:
    """
    Execute K-means algorithm on image dataset
    :param images: Image data
    :param normalise: Boolean to determine if normalisation needs to occur
    :return: Dictionary of new clusters, centers and data
    """
    x_data, _ = separate_x_and_y(images)  # Extracts x data from image
    if normalise:
        x_data = x_data / 255
    k = find_best_k(x_data)
    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(x_data)
    cluster_centers = kmeans.cluster_centers_
    clusters = get_clustered_images(x_data, y_pred)
    inertia = kmeans.inertia_
    results = {'k': k, 'cluster': y_pred, 'center': cluster_centers, 'data': clusters, 'inertia': inertia}
    return results


def main(sys_args):
    """"
    This is very slow. It is slow because it collects the data of 5 different k means runs and then prints it.
    K means is very computationally taxing.
    """
    images = load_image_data("../data/x_train_gr_smpl_40.csv", "../data/y_train_smpl.csv")  # Used for naive bayes
    shuffled_images = shuffle_dataframe(images)
    _, original_y = separate_x_and_y(shuffled_images)
    original_y.columns = ['original']
    accuracy = {}
    for i in range(0, 5):
        result = k_means_clustering(images=shuffled_images, normalise=True)
        accuracy[i] = accuracy_of_clusters(y_data=original_y, new_groups=result['cluster'])
        accuracy[i]['inertia'] = result['inertia']
    for k in range(len(accuracy.keys())):
        print(accuracy[k])

    # for i in range(len(clusters)):
    #     cluster_to_csv(clusters[i], i, normalise)
    # plot_entire_clusters(clusters, cluster_centers, normalised=normalise, show=False)

    # path = Path("../evidence/K-Means")
    # if path.is_dir():
    #     if normalise:
    #         full_path = Path(f'{path}/inertia_nomalised.txt')
    #     else:
    #         full_path = Path(f'{path}/inertia_non-normalised.txt')
    #     with open(full_path, 'a') as f:
    #         f.write(f"{kmeans.inertia_}\n")

    # parallel_coord_plot(x_data, y_pred, show=False, normalised=normalise)
    # coord_plot(x_data, y_pred, k, False)


if __name__ == "__main__":
    main(sys.argv)
