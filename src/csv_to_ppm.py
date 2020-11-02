import argparse
import math
import sys


def csv_to_ppm(path: str, csv_line: str):
    """
    Coverts the given csv_line (representing an image) into P2 PPM image file.
    :param path: Path where the PPM image will be saved
    :param csv_line: Line of comma separated pixels in text form
    """
    pixels = csv_line.split(',')
    image_size = math.isqrt(len(pixels))
    with open(path, 'w+') as file:
        file.write('P2\n')  # PGM, greyscale image saved as text
        file.write(f'{image_size} {image_size}\n')  # Width and height
        file.write('255\n')  # Maximum pixel value
        # Write the pixels, because the format specifies that the line should
        # not exceed 70 characters, we will write just one pixel per line
        # (even though that's only max 4 characters (3 digits + \n))
        file.write(' \n'.join([num[0:num.index('.')] for num in pixels]))  # Will remove everything after the dot


def csv_to_ppm_from_file(original_file: str, image_index: int, target_file: str):
    """
    Coverts an image from the CSV file into P2 PPM image file
    :param original_file: Original CSV file with images
    :param image_index: Index of the image to be converted
    :param target_file: Filename of the PPM image
    """
    with open(original_file, 'r') as file:
        file.readline()  # Skip the first header line
        for i, line in enumerate(file):
            if i == image_index:
                csv_to_ppm(target_file, line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Coverts CSV image into PPM')
    parser.add_argument('original_file', type=str, help='Original file with CSV images')
    parser.add_argument('image_index', type=int, help='Index of the image to be converted into PPM')
    parser.add_argument('target_file', type=str, help='Filename of the PPM image')
    args = parser.parse_args(sys.argv[1:])

    csv_to_ppm_from_file(args.original_file, args.image_index, args.target_file)
