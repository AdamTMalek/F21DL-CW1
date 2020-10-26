from image_cropper import crop_images
from get_dataframes import load_image_data, shuffle_dataframe
from naive_bayes import naive_bayes
import pandas as pd


def main():
    crop_images("data/x_train_gr_smpl.csv", 40)
    images = load_image_data(
        "data/x_train_gr_smpl_40.csv", "data/y_train_smpl.csv")
    shuffled_images = shuffle_dataframe(images)
    naive_bayes(shuffled_images)


if __name__ == "__main__":
    main()
