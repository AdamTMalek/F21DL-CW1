from sklearn import mixture
from sklearn.model_selection import train_test_split
from sklearn import metrics
from dataframe_manipulation import separate_x_and_y
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classes import classes
from dataframe_manipulation import load_image_data, shuffle_dataframe, load_image_data_with_ten_one

def select_best(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]

# Based on code from https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4 by Vincenzo Lavorini
def find_best_component_count(images):
    '''
    Calculates the silhouette score for each possible number of clusters (2 - 20)
    Plots this data on a graph 
    Note: Due to the number of iterations this algorithm runs through, it can take hours to run

    '''
    x_data, _ = separate_x_and_y(images)  # Extracts x data from image

    #Range of possible cluster counts
    n_clusters=np.arange(2, 20)
    sils=[]
    sils_err=[]

    #Number of iterations per possible cluster count
    iterations=20

    #Loop through all possible number of clusters
    for n in n_clusters:
        tmp_sil=[]
        print("Starting iteration ", n)

        #Iterate for cluster count
        for _ in range(iterations):
            #Create Gaussian mixture model
            gmm=mixture.GaussianMixture(n, n_init=5)
            #Fit and predict data
            labels=gmm.fit_predict(x_data)
            #Calculate silhouette score for given x_data and labels using euclidean distance
            sil=metrics.silhouette_score(x_data, labels, metric='euclidean')
            #Add to temporary silhouette score list
            tmp_sil.append(sil)
        
        #Calculate the mean value from the best in the temporary silhouette score list
        val=np.mean(select_best(np.array(tmp_sil), int(iterations/5)))
        #Calculate standard deviation for use as error bar
        err=np.std(tmp_sil)
        #Append to lists
        sils.append(val)
        sils_err.append(err)
        print("Done iteration ", n)

    plt.errorbar(n_clusters, sils, yerr=sils_err)
    plt.title("Silhouette Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters")
    plt.ylabel("Score")

def em_clustering(images):
    for i in range(2, 15):
        x_data, y_data = separate_x_and_y(images)  # Extracts x data from image

        #
        gmm = mixture.GaussianMixture(
            n_components=i, covariance_type='full', n_init=3)

        y_pred = gmm.fit_predict(x_data)

        #Yes you could almost definitely do this better
        correct = 0
        for index, row in y_data.iterrows():
            if y_pred[index] == row.iloc[0]:
                correct += 1

        accuracy = (correct / len(y_data)) * 100
        print("Num: ", i, " - Accuracy: ", accuracy, "%   - Dataset Length: ", len(y_data))


images = load_image_data(
    "c:/Users/bruce/Desktop/Education/Uni Work/F20DL/F21DL-CW1/data/x_train_gr_smpl_pruned_20.csv", "c:/Users/bruce/Desktop/Education/Uni Work/F20DL/F21DL-CW1/data/y_train_gr_smpl_pruned_20.csv") 
shuffled_images = shuffle_dataframe(images)

em_clustering(images)
