from get_dataframes import load_image_data_with_ten_one
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


images = load_image_data_with_ten_one()
corr_matrix = images.corr()
ten_highest_corr_20 = corr_matrix["Speed Limit 20"].sort_values(
    ascending=False)

print("Speed 20 Correlation:")
print(ten_highest_corr_20[:10])
