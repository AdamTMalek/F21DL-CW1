import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import urllib
import tarfile
import os
import numpy as np
import sklearn
import sys
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"
# To plot pretty figures


# Loads the image and concatenates with the y-data holding class data from 0-9
def load_image_data():
    x_data = pd.read_csv("data/x_train_gr_smpl.csv")
    y_data = pd.read_csv("data/y_train_smpl.csv", names=["Class"])
    concatenated = pd.concat([x_data, y_data], axis=1)
    return concatenated


# Loads all the boolean y-data files, concatenates and returns all of them
def load_all_y_data():
    data = pd.DataFrame()
    for x in range(0, 9):
        y = pd.read_csv("data/y_train_smpl_"+str(x)+".csv", header=0)
        y.columns = ["Class " + str(x)]
        data = pd.concat([data, y], axis=1)
    return data


# Loads the image and concatenates with each boolean y-data file, 0=true 1=false
def load_image_data_with_ten_one():
    x_data = pd.read_csv("data/x_train_gr_smpl.csv")
    y_data = load_all_y_data()
    concatenated = pd.concat([x_data, y_data], axis=1)
    return concatenated
