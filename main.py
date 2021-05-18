import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from PIL import Image
import numpy as np
from util import Const
from model import Model


def get_args():
    parser = argparse.ArgumentParser(
        description='Soil assessment tool',
        add_help=True,
    )

    parser.add_argument(
        '-f',
        '--file',
        required=True,
        action='store',
        type=str,
        help='Path to the image to perform soil assessment',
    )

    return validate_args(parser.parse_args())


def validate_args(args):
    args.file = os.path.abspath(args.file)

    if not os.path.exists(args.file) or not os.path.isfile(args.file):
        raise FileNotFoundError(f'File on path "{args.file}" does not not exist or is not a file')

    return args


def get_image(args):
    image: Image.Image = Image.open(args.file)

    ideal_ratio = Const.YANDEX_MAPS_MAX_WIDTH / Const.YANDEX_MAPS_MAX_HEIGHT

    width, height = image.size
    ratio = width / height

    if abs(ideal_ratio - ratio) > 1e-4:
        if ratio - ideal_ratio > 1e-4:  # image is wider than needed
            # perform a central crop along width axis
            new_width = ideal_ratio * height
            difference = width - new_width
            image = image.crop((difference // 2, 0, width - difference // 2, height))
        else:  # image is taller than needed
            # perform a central crop along height axis
            new_height = width / ideal_ratio
            difference = height - new_height
            image = image.crop((0, difference // 2, width, height - difference // 2))

    image = image.resize((Const.YANDEX_MAPS_MAX_WIDTH, Const.YANDEX_MAPS_MAX_HEIGHT))

    return np.array(image)


def main():
    image = get_image(get_args())

    model = Model()
    pred = model.predict([image])
    pred = pred[0][0]

    print(f'Organic matter concentration on given image is {pred:0.3%}')


if __name__ == '__main__':
    main()
