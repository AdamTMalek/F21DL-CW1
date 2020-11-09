from sklearn.model_selection import train_test_split
from sklearn import metrics
from dataframe_manipulation import separate_x_and_y
from sklearn.cluster import AffinityPropagation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classes import classes
from dataframe_manipulation import load_image_data, shuffle_dataframe, load_image_data_with_ten_one

#https://www.ritchievink.com/blog/2018/05/18/algorithm-breakdown-affinity-propagation/
#https://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html#sphx-glr-auto-examples-cluster-plot-affinity-propagation-py
def ap_clustering(images):
    x_data, y_data = separate_x_and_y(images)  # Extracts x data from image
    #random_state=5,
    #Create Affinity Propagation class
    af = AffinityPropagation(preference=-800*65025,  damping=0.9, max_iter=1000)
    y_pred = af.fit_predict(x_data)
    
    #Calculate how many predicted classes are correct classes
    correct = 0
    for index, row in y_data.iterrows():
        if y_pred[index] == row.iloc[0]:
            correct += 1

    #Print output data
    accuracy = (correct / len(y_data)) * 100
    print("Top 5 - Damping: 0.9")
    print("Num: ", af.preference, " - Accuracy: ", accuracy, "%   - Dataset Length: ", len(y_data))
    print("Iterations: ", af.n_iter_, " Centres: ", af.cluster_centers_ , " Number of clusters: ", len(af.cluster_centers_))


#Load image data
images = load_image_data(
    "c:/Users/bruce/Desktop/Education/Uni Work/F20DL/F21DL-CW1/data/x_train_gr_smpl_pruned_5.csv", "c:/Users/bruce/Desktop/Education/Uni Work/F20DL/F21DL-CW1/data/y_train_gr_smpl_pruned_20.csv") 

ap_clustering(images)
