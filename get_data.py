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


def load_image_data():
    x_data = pd.read_csv("data/x_train_gr_smpl.csv")
    y_data = pd.read_csv("data/y_train_smpl.csv", names=["Class"])
    concatenated = pd.concat([x_data, y_data], axis=1)
    return concatenated


