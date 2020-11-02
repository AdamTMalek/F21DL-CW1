import argparse
import itertools
import random
import sys
from typing import List, Dict

NUMBER_OF_CLASSES = 10
MAX_NUMBER_OF_IMAGES_PER_CLASS = 210
NUMBER_OF_IMAGES = {
    0: 210,
    1: 2220,
    2: 2250,
    3: 1410,
    4: 1980,
    5: 210,
    6: 360,
    7: 240,
    8: 540,
    9: 270
}


def reduce_data(original_file: str, target_file: str, images_per_class: int, pick_random: bool = False):
    """
    Reduces the data set by taking the specified number of images per class.
    :param original_file: Original data set, with all the images.
    :param target_file: Target (reduced) data set
    :param images_per_class: Number of images per class to take
    :param pick_random: Should the algorithm pick random numbers. If false, it will take the first images.
    """

    def get_indices(class_number: int) -> List[int]:
        # First, find at which line does the class start
        start = sum([NUMBER_OF_IMAGES[number] for number in range(0, class_number)])

        if pick_random:
            # Pick random indices (line numbers) in the range start -> start + number of images belonging to that class.
            # The random numbers will be uniformly distributed.
            random_indices = random.sample(range(start, start + NUMBER_OF_IMAGES[class_number]), images_per_class)
        else:
            random_indices = [_ for _ in range(start, start + images_per_class)]
        return random_indices

    def find_key_for_value(dictionary: Dict[int, int], value: int) -> int:
        for k, v in dictionary.items():
            if value in v:
                return k

    # Because our smallest class has MAX_NUMBER_OF_IMAGES_PER_CLASS images
    # and the algorithm takes equal amount of images from each class
    # we have to check images_per_class is within the acceptable range
    if images_per_class > MAX_NUMBER_OF_IMAGES_PER_CLASS:
        raise ValueError(f'Images per class cannot be greater than {MAX_NUMBER_OF_IMAGES_PER_CLASS}')

    # Dictionary where keys are class numbers and values are lists of line numbers to take
    indices = {
        class_number: get_indices(class_number)
        for class_number in range(0, NUMBER_OF_CLASSES)
    }

    with open(original_file, 'r') as original_data_set:
        with open(target_file, 'w+') as reduced_data_set:
            # Write the header line, add the class attribute which doesn't exist in the original file
            reduced_data_set.write(original_data_set.readline().rstrip('\n') + ',class\n')

            for i, line in enumerate(original_data_set):
                if i in list(itertools.chain.from_iterable([v for _, v in indices.items()])):
                    class_num = find_key_for_value(indices, i)
                    reduced_data_set.write(line.rstrip('\n') + f',{class_num}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reduces the data set')
    parser.add_argument('original_file', type=str, help='Original file with CSV images')
    parser.add_argument('target_file', type=str, help='Filename of the reduced data set')
    parser.add_argument('images_per_class', type=int, help='Number of images to take per class')
    parser.add_argument('--random', type=bool, help='True if to pick random images')
    args = parser.parse_args(sys.argv[1:])

    reduce_data(args.original_file, args.target_file, args.images_per_class, args.random)
