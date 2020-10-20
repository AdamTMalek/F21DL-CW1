from get_dataframes import load_image_data_with_ten_one
from image_cropper import crop_images
import pandas as pd
import numpy as np
import sklearn
import sys
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"

# WILL TAKE TIME TO EXECUTE!
classes = ["Speed Limit 20", "Speed Limit 30", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
           "Left Turn", "Right Turn", "Beware Pedestrian Crossing", "Beware Children", "Beware Cycle Route Ahead"]


# Returns the top i pixels of col i from images dataframe
def get_top_corr_pixels(images, col, i):
    corr_matrix = images.corr()
    highest_corr = corr_matrix[col].sort_values(
        ascending=False)
    # from 1 to i because the 0th element is the class attribute
    return highest_corr[1:i].index.values


# Returns the image dataframe with only the top i correlating pixels for each class
def remove_none_corr_pixels(images, i):
    number_of_cols = 5
    pruned_images = pd.DataFrame()
    top_pixels = []
    for class_tag in classes:
        top_pixels = np.append(
            top_pixels, get_top_corr_pixels(images, class_tag, i))
    print(top_pixels)
    images.drop(images.columns.difference(top_pixels), 1, inplace=True)
    print(sorted(top_pixels, key=int))
    return images
