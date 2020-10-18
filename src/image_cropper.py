import os
import sys
import math
import argparse


def crop_image(line: str, new_size: int) -> str:
    """
    Crops a single image represented as a single line of comma separated values,
    and returns the cropped as a single line of comma separated values.

    The algorithm assumes square images (width=height)
    :param line: Original image (csv)
    :param new_size: New size (width=height)
    :return: Cropped image (csv)
    """
    values = [value.lstrip() for value in line.split(',')]  # Split the values into a list. Remove leading spaces.
    original_size = math.sqrt(len(values))
    crop_size = (original_size - new_size) // 2  # Amount of pixels that will be cropped from each side

    rows_to_crop = int(crop_size * original_size)
    values = values[rows_to_crop:-rows_to_crop]  # Crop the top and bottom of the image

    cropped_line = ''
    for index, value in enumerate(values):
        current_width = index % original_size
        if current_width < crop_size or current_width >= (original_size - crop_size):
            continue  # Crop the sides
        cropped_line += value + ', '
    return cropped_line[:-2]  # Remove trailing comma and space at the end of the line


def crop_images(original_file_path: str, new_size: int) -> None:
    """
    Crops all images in the passed CSV file to match the given size,
    then saves them to a new file with the name [original_file_path]_[new_size].csv

    The algorithm assumes square images (width=height)
    :param original_file_path: Path to the file with images that are to be cropped
    :param new_size: New size (width=height)
    """
    filename = os.path.splitext(original_file_path)[0]
    cropped_images_file = f'{filename}_{new_size}.csv'

    with open(original_file_path, 'r') as original_file:
        with open(cropped_images_file, 'w') as new_file:
            next(original_file)  # Skip headers line
            for line in original_file:
                cropped_line = crop_image(line, new_size)
                new_file.write(cropped_line)


def main(sys_args):
    parser = argparse.ArgumentParser(description="Crops the images")
    parser.add_argument('original_file', type=str, help='CSV File with original images')
    parser.add_argument('new_size', type=int, help='New size (width and height)')
    args = parser.parse_args(sys_args)
    print(args)


if __name__ == "__main__":
    main(sys.argv)
