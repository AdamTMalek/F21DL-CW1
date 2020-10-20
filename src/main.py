from image_cropper import crop_images
from get_dataframes import load_image_data_with_ten_one, shuffle_dataframe
from get_top_correlating_pixels import remove_none_corr_pixels


def main():
    crop_images("../data/x_train_gr_smpl.csv", 40)
    #images = load_image_data_with_ten_one("../data/x_train_gr_smpl_40.csv")
    # shuffle_dataframe(images)
    #pruned_images = remove_none_corr_pixels(images, 10)
    # print(pruned_images)


if __name__ == "__main__":
    main()
