from classes import classes
from dataframe_manipulation import load_image_data_with_ten_one
from image_cropper import crop_images
import pandas as pd
import numpy as np
import sklearn
import sys
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"

# WILL TAKE TIME TO EXECUTE!

# Returns the top i pixels of col i from images dataframe


def get_top_corr_pixels_for_col(images, col, i):
    corr_matrix = images.corr()
    highest_corr = corr_matrix[col].sort_values(
        ascending=False)
    # from 1 to i because the 0th element is the class attribute
    print("The top", i, "correlating pixels for col", col, "are:")
    print(highest_corr[1:i+1].index.values)
    return highest_corr[1:i+1].index.values


def get_top_corr_pixels(images, i):
    top_pixels = []
    for class_tag in classes:
        top_pixels = np.append(
            top_pixels, get_top_corr_pixels_for_col(images, class_tag, i))
    return top_pixels


# Returns the image dataframe with only the top i correlating pixels for each class
def remove_none_corr_pixels(images, top_pixels):
    pruned_images = images.copy()
    # Append 0 as the column label for classes is 0
    top_pixels = np.append(top_pixels, '0')
    pruned_images.drop(pruned_images.columns.difference(
        top_pixels), 1, inplace=True)
    return pruned_images
