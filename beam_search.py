import cv2
from tqdm import tqdm
from typing import List
import random
from painting import Painting
from utils import (PICTURE_SIZE, create_gif,
                   save_images_to_folder, extract_features, random_rectangle_set)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


RECTANGLE_AMOUNT = 300
RANDOM_TILES_PER_ROUND = 20
BEAM_WIDTH = 10
ITER_NUM = 1000
MAX_SIZE = 20
MULTIPLY_WEIGHTS = 50
COLOR_ME_TENDERS = True
SSIM_WEIGHT = 0.5
FEATURE_EXTRACT = True
LOSS_TYPE = 'deltassim'

IMAGE_NAME = 'FELV-cat'
directory = f'./beam_search/{IMAGE_NAME}/{RECTANGLE_AMOUNT}_{RANDOM_TILES_PER_ROUND}_{BEAM_WIDTH}_{ITER_NUM}_' \
            f'{MAX_SIZE}_{MULTIPLY_WEIGHTS}_{SSIM_WEIGHT}_{COLOR_ME_TENDERS}_{LOSS_TYPE}'


class BeamSearch_Node(Painting):
    def __init__(self, rectangle_list, target_img, weights_matrix, loss_type=LOSS_TYPE,
                 feature_extract=FEATURE_EXTRACT, ssim_weight=SSIM_WEIGHT,
                 parent=None):
        super().__init__(rectangle_list, target_img, weights_matrix, loss_type, feature_extract, ssim_weight)
        self.parent = parent

    def get_path(self) -> List:
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]  # return reversed path


def random_init_canvas(target_img) -> BeamSearch_Node:
    """
    creates a canvas that was randomly initialized
    """
    rectangle_list = random_rectangle_set(number_of_rectangles=RECTANGLE_AMOUNT, target_image=target_img,
                                          max_size=MAX_SIZE, edge_thickness=0, color_random=COLOR_ME_TENDERS)
    weights_matrix = extract_features(target_img, MULTIPLY_WEIGHTS)
    return BeamSearch_Node(rectangle_list, target_img, weights_matrix)


def beam_search(target_img, n_brushstrokes=RECTANGLE_AMOUNT, beam_width=BEAM_WIDTH,
                n_candidates=RANDOM_TILES_PER_ROUND, iterations=ITER_NUM):
    """
    performs a bram search and returns a list of blocks that represent the painting
    """
    #  add color options(delta e from genetic boi),
    #  and after it works implement the SSIM loss

    init_node = random_init_canvas(target_img)
    beam = [init_node]

    for _ in tqdm(range(iterations)):
        new_beam: List[BeamSearch_Node] = []

        for node in beam:
            rectangle_list = node.rectangle_list
            index_to_modify = random.randint(0, n_brushstrokes - 1)
            new_tiles = random_rectangle_set(n_candidates, target_img, MAX_SIZE, 0, COLOR_ME_TENDERS)

            for new_tile in new_tiles:
                new_rectangle_list = rectangle_list.copy()
                new_rectangle_list[index_to_modify] = new_tile
                new_canvas = BeamSearch_Node(new_rectangle_list, target_img, init_node.weights_matrix, parent=node)
                new_beam.append(new_canvas)

        beam = sorted(new_beam)[:beam_width]

    return beam[0]


if __name__ == '__main__':
    original_image_path = f'layouts/{IMAGE_NAME}.jpg'
    target_img = cv2.imread(original_image_path)
    target_img = cv2.resize(target_img, PICTURE_SIZE)

    final_node = beam_search(target_img=target_img, n_brushstrokes=RECTANGLE_AMOUNT, beam_width=BEAM_WIDTH,
                             n_candidates=RANDOM_TILES_PER_ROUND, iterations=ITER_NUM)

    path = final_node.get_path()  # Get the path from the final node

    save_images_to_folder(directory+"/best_image_path", [node.image for node in path], target_img,
                          [f"Iteration: {i}" for i in range(len(path))])
    create_gif(directory+"/best_image_path", directory + "/GIF.gif")

    losses = [node.loss for node in path]
    plt.figure(figsize=(15, 7))  # Set the figure size
    plt.plot(losses, linestyle='-', color='blue')
    plt.title('Loss per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(directory + "/loss.jpg")