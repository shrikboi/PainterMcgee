import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Tuple
import random
from utils import Rectangle, draw_rectangle, PICTURE_SIZE, calculate_color, display_images_side_by_side

RANDOM_TILES_PER_ROUND = 20
NUM_OF_BRUSHSTROKES = 500
# NUM_OF_TILES = 1000
BEAM_WIDTH = 5
ITER_NUM = 1000
RECTANGLE_LIST = './layouts/rectangle_list.txt'
IMAGE_FILENAME = './layouts/FELV-cat.jpg'


def load_rectangle_list(filename):
    rectangles = []
    with open(filename, 'r') as f:
        for line in f:
            width, height, angle, edge_thickness = map(int, line.strip().split())
            rectangles.append((width, height, angle, edge_thickness))
    return rectangles


def get_empty_canvas(target_img) -> np.ndarray:
    """
    returns an empty(white) canvas the size of the target image
    """
    return np.ones((PICTURE_SIZE[1], PICTURE_SIZE[0], 3), dtype=np.uint8) * 255


def create_random_tile(rectange_list: List, target_image) -> Rectangle:
    """
    generates random tile with x,y location
    """

    width, height, angle, edge_thickness = random.choice(rectange_list)
    center_x = random.randint(0, PICTURE_SIZE[0] - 1)
    center_y = random.randint(0, PICTURE_SIZE[1] - 1)
    color = calculate_color((width, height), (center_x, center_y), angle, target_image)
    opacity = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    return Rectangle((width, height), angle, edge_thickness, (center_x, center_y), color, opacity)


def get_tile_set(rectangle_list: List, target_image, set_size: int):
    return [create_random_tile(rectangle_list, target_image) for _ in range(set_size)]


def apply_brushstrokes(canvas, brushstrokes):
    for stroke in brushstrokes:
        canvas = draw_rectangle(canvas.copy(), stroke)
    return canvas


def mse_loss(image, target_image) -> float:
    return np.sum((image.astype(np.float32) - target_image.astype(np.float32)) ** 2)

def evaluate_canvas():
    pass


def random_init_canvas(target_image, rectangle_list) -> Tuple:
    """
    creates a canvas that was randomly initialized
    """
    canvas = get_empty_canvas(target_image)
    stroke_set = get_tile_set(rectangle_list, target_image, NUM_OF_BRUSHSTROKES)

    canvas = apply_brushstrokes(canvas, stroke_set)
    return canvas, stroke_set


def beam_search(target_image, rectangle_list, n_brushstrokes=NUM_OF_BRUSHSTROKES, beam_width=BEAM_WIDTH,
                n_candidates=RANDOM_TILES_PER_ROUND, iterations=ITER_NUM):
    """
    performs a bram search and returns a list of blocks that represent the painting
    """
    # setup random init canvas
    # run iterations
    # get random tileset
    # get tile to replace
    # replace with the best option

    init_canvas, init_stroke_set = random_init_canvas(target_image, rectangle_list)
    init_loss = mse_loss(init_canvas, target_image)
    beam = [(init_canvas, init_stroke_set, init_loss)]

    for _ in tqdm(range(iterations)):
        new_beam = []

        for _, stroke_set, _ in beam:
            index_to_modify = random.randint(0, n_brushstrokes - 1)
            new_tiles = get_tile_set(rectangle_list, target_image, n_candidates)

            for new_tile in new_tiles:
                new_stroke_set = stroke_set.copy()
                new_stroke_set[index_to_modify] = new_tile

                new_canvas = get_empty_canvas(target_image)
                new_canvas = apply_brushstrokes(new_canvas, new_stroke_set)

                new_loss = mse_loss(new_canvas, target_image)
                new_beam.append((new_canvas, new_stroke_set, new_loss))

        beam = sorted(new_beam, key=lambda x: x[2])[:beam_width]

    return beam[0]

    # cv2.imshow('Initial Canvas', init_canvas)
    # cv2.waitKey(0)


if __name__ == '__main__':
    target_image = cv2.imread(IMAGE_FILENAME)
    target_image = cv2.resize(target_image, PICTURE_SIZE)

    rectangle_list = load_rectangle_list(RECTANGLE_LIST)

    final_canvas, final_stroke_set, final_loss = beam_search(
        target_image,
        rectangle_list,
        n_brushstrokes=NUM_OF_BRUSHSTROKES,
        beam_width=BEAM_WIDTH,
        n_candidates=RANDOM_TILES_PER_ROUND,
        iterations=ITER_NUM
    )
    result_image = display_images_side_by_side(target_image, final_canvas)
    cv2.imshow("Target vs Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



