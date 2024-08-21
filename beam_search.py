import os
import cv2

import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import random
import time
import abc

import colour

import skimage as ski
from utils import (draw_rectangle, PICTURE_SIZE, generate_random_rectangle, create_gif, display_images_side_by_side,
                   display_image, extract_features)

RANDOM_TILES_PER_ROUND = 10
NUM_OF_BRUSHSTROKES = 120
BEAM_WIDTH = 5
ITER_NUM = 1000
RECTANGLE_LIST = './layouts/rectangle_list.txt'
IMAGE_FILENAME = './layouts/FELV-cat.jpg'
MAX_SIZE = 20
OUTPUT_FOLDER = './output_paintings'
OUTPUT_GIF = './output_gif.gif'
MULTIPLY_WEIGHTS = 50


class Node:
    def __init__(self, canvas, stroke_set, loss, parent=None):
        self.canvas = canvas
        self.stroke_set = stroke_set
        self.loss = loss
        self.parent = parent

    def get_path(self) -> List:
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]  # return reversed path


def get_empty_canvas() -> np.ndarray:
    """
    returns an empty(white) canvas the size of the target image
    """
    return np.ones((PICTURE_SIZE[1], PICTURE_SIZE[0], 3), dtype=np.uint8) * 255


def get_tile_set(tgt_image, set_size: int,  max_block_size, radomize_color=True):
    tiles = []
    for _ in range(set_size):
        tiles.append(generate_random_rectangle(tgt_image, max_block_size, 0, color_random=radomize_color))
    return tiles


def apply_brushstrokes(canvas, brushstrokes):
    for stroke in brushstrokes:
        canvas = draw_rectangle(canvas.copy(), stroke)
    return canvas


def mse_loss(image, tgt_image) -> float:
    return np.sum((image.astype(np.float32) - tgt_image.astype(np.float32)) ** 2)


def delta_e_loss(image, image2, weights_matrix):
    image1 = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    image2 = cv2.cvtColor(image2.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)

    delta_e = colour.difference.delta_e.delta_E_CIE2000(image1, image2)
    delta_e = delta_e * weights_matrix
    return np.mean(delta_e)


def random_init_canvas(tgt_image) -> Tuple:
    """
    creates a canvas that was randomly initialized
    """
    canvas = get_empty_canvas()
    stroke_set = get_tile_set(tgt_image, NUM_OF_BRUSHSTROKES, MAX_SIZE)

    canvas = apply_brushstrokes(canvas, stroke_set)
    return canvas, stroke_set


def save_paintings_to_folder(directory, node_list: List[Node], side_by_side=False, og_photo=None):
    """
    Save the paintings represented by the nodes in the beam search to a folder.

    :param directory: The directory where the images will be saved.
    :param node_list: List of Node objects, each representing a painting state.
    :param side_by_side: Boolean indicating whether to display the original photo alongside the generated painting.
    :param og_photo: The original photo to display alongside the painting if side_by_side is True.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, node in enumerate(node_list):
        painting = np.ones((*PICTURE_SIZE[::-1], 3), dtype=np.uint8) * 255
        painting = apply_brushstrokes(painting, node.stroke_set)

        if side_by_side and og_photo is not None:
            combined_image = display_images_side_by_side(og_photo, painting)
            cv2.imwrite(os.path.join(directory, f"image_{i}.png"), combined_image)
        else:
            cv2.imwrite(os.path.join(directory, f"image_{i}.png"), display_image(painting))


def beam_search(tgt_image, weight_matrix,  n_brushstrokes=NUM_OF_BRUSHSTROKES, beam_width=BEAM_WIDTH,
                n_candidates=RANDOM_TILES_PER_ROUND, iterations=ITER_NUM):
    """
    performs a bram search and returns a list of blocks that represent the painting
    """
    #  add color options(delta e from genetic boi),
    #  and after it works implement the SSIM loss

    init_canvas, init_stroke_set = random_init_canvas(tgt_image)
    init_loss = mse_loss(init_canvas, tgt_image)
    # init_loss = delta_e_loss(init_canvas, tgt_image, weights_matrix)
    beam = [Node(init_canvas, init_stroke_set, init_loss)]

    # cv2.imshow('Initial Canvas', init_canvas)
    # cv2.waitKey(0)

    cv2.imwrite('./init_image.jpeg', init_canvas)

    for _ in tqdm(range(iterations)):
        new_beam: List[Node] = []

        for node in beam:
            stroke_set = node.stroke_set
            index_to_modify = random.randint(0, n_brushstrokes - 1)
            new_tiles = get_tile_set(tgt_image, n_candidates, MAX_SIZE)

            for new_tile in new_tiles:
                new_stroke_set = stroke_set.copy()
                new_stroke_set[index_to_modify] = new_tile

                new_canvas = get_empty_canvas()
                new_canvas = apply_brushstrokes(new_canvas, new_stroke_set)

                # new_loss = mse_loss(new_canvas, tgt_image)
                new_loss = delta_e_loss(init_canvas, tgt_image, weights_matrix)
                new_node = Node(new_canvas, new_stroke_set, new_loss, parent=node)
                new_beam.append(new_node)

        beam = sorted(new_beam, key=lambda x: x.loss)[:beam_width]

    return beam[0]


if __name__ == '__main__':
    target_image = cv2.imread(IMAGE_FILENAME)
    target_image = cv2.resize(target_image, PICTURE_SIZE)

    weights_matrix = extract_features(target_image, MULTIPLY_WEIGHTS)

    final_node = beam_search(
        target_image,
        weights_matrix,
        n_brushstrokes=NUM_OF_BRUSHSTROKES,
        beam_width=BEAM_WIDTH,
        n_candidates=RANDOM_TILES_PER_ROUND,
        iterations=ITER_NUM
    )

    path = final_node.get_path()  # Get the path from the final node

    save_paintings_to_folder(OUTPUT_FOLDER, path, side_by_side=True, og_photo=target_image)
    create_gif(OUTPUT_FOLDER, OUTPUT_GIF)

    # result_image = display_images_side_by_side(target_image, final_node.canvas)
    # cv2.imshow("Target vs Result", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
