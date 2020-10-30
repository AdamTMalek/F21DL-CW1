from image_cropper import crop_images
from dataframe_manipulation import load_image_data, shuffle_dataframe, load_image_data_with_ten_one
from naive_bayes import naive_bayes
from get_top_correlating_pixels import get_top_corr_pixels, remove_none_corr_pixels

top_corr_pixels = [5, 10, 20]


def main():
    #crop_images("data/x_train_gr_smpl.csv", 40)
    images = load_image_data(
        "data/x_train_gr_smpl_40.csv", "data/y_train_smpl.csv")  # Used for naive bayes
    corr_images = load_image_data_with_ten_one(
        "data/x_train_gr_smpl_40.csv")  # Used for finding top correlating pixels

    shuffled_images = shuffle_dataframe(images)
    shuffled_corr_images = shuffle_dataframe(corr_images)
    print("=========== Unpruned ================")
    naive_bayes(shuffled_images)
    for x in top_corr_pixels:
        print("=========== Top", x, "pixels ================")
        top_pixels = get_top_corr_pixels(shuffled_corr_images, x)
        pruned_images = remove_none_corr_pixels(shuffled_images, top_pixels)
        naive_bayes(pruned_images)


if __name__ == "__main__":
    main()
