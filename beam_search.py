import cv2
from tqdm import tqdm
from typing import List
import random
from painting import BeamSearch_Node, LOSS
from utils import (PICTURE_SIZE, create_gif,
                   save_images_to_folder, extract_features, random_rectangle_set)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Constants defining the beam search parameters
RECTANGLE_AMOUNT = 350 # Number of rectangles in each painting
RANDOM_TILES_PER_ROUND = 10 # Number of random rectangles to generate per round
BEAM_WIDTH = 5 # Number of best nodes to keep in the beam
ITER_NUM = 2000 # Number of iterations for the beam search
LOSS_TYPE = LOSS.MSESSIM # Type of loss function to use
IMAGE_NAME = 'FELV-cat' # Name of the target image

MULTIPLY_WEIGHTS = 50
COLOR_ME_TENDERS = True # if true color randomly chosen, else color is chosen by average color in target rectangle area
EDGE_THICKNESS = 0 # Thickness of the rectangle edges
MAX_SIZE = 20 # Maximum size of rectangles

# Directory to save generated images
directory = f'./beam_search/{IMAGE_NAME}/{RECTANGLE_AMOUNT}_{RANDOM_TILES_PER_ROUND}_{BEAM_WIDTH}_{ITER_NUM}_' \
            f'{MAX_SIZE}_{MULTIPLY_WEIGHTS}_{COLOR_ME_TENDERS}_{LOSS_TYPE}'


def random_init_canvas(target_img):
    """
    Initialize a canvas randomly by generating a set of rectangles that approximate the target image.
    @param target_img: The target image to be approximated.
    @return: A BeamSearch_Node object representing the initial state of the canvas.
    """
    rectangle_list = random_rectangle_set(number_of_rectangles=RECTANGLE_AMOUNT, target_img=target_img,
                                          max_size=MAX_SIZE, edge_thickness=EDGE_THICKNESS, color_random=COLOR_ME_TENDERS)
    weights_matrix = extract_features(target_img, MULTIPLY_WEIGHTS)
    return BeamSearch_Node(rectangle_list, target_img, weights_matrix, LOSS_TYPE)


def beam_search(target_img, n_brushstrokes=RECTANGLE_AMOUNT, beam_width=BEAM_WIDTH,
                n_candidates=RANDOM_TILES_PER_ROUND, iterations=ITER_NUM):
    """
    Perform a beam search to find the best sequence of rectangles that approximate the target image.
    @param target_img: The target image to be approximated.
    @param n_brushstrokes: The number of rectangles in the painting.
    @param beam_width: The number of top candidates to keep at each step.
    @param n_candidates: The number of random rectangles generated in each round.
    @param iterations: The number of iterations for the beam search.
    @return: The best BeamSearch_Node found after all iterations.
    """
    init_node = random_init_canvas(target_img)
    beam = [init_node]

    for _ in tqdm(range(iterations)):
        new_beam: List[BeamSearch_Node] = []

        for node in beam:
            rectangle_list = node.rectangle_list
            index_to_modify = random.randint(0, n_brushstrokes - 1) # Randomly choose a rectangle to modify
            new_tiles = random_rectangle_set(n_candidates, target_img, MAX_SIZE, EDGE_THICKNESS, COLOR_ME_TENDERS)

            # Create new candidate nodes by modifying the chosen rectangle
            for new_tile in new_tiles:
                new_rectangle_list = rectangle_list.copy()
                new_rectangle_list[index_to_modify] = new_tile
                new_canvas = BeamSearch_Node(new_rectangle_list, target_img, init_node.weights_matrix, LOSS_TYPE,
                                             parent=node)
                new_beam.append(new_canvas)

        # Keep only the top `beam_width` candidates
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
                          bottom_texts=[f"Iteration: {i}" for i in range(len(path))])
    create_gif(directory+"/best_image_path", directory + "/GIF.gif")

    losses = [node.loss for node in path]
    plt.figure(figsize=(8, 7))  # Set the figure size
    plt.plot(losses, linestyle='-', color='blue')
    plt.title('Loss per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(directory + "/loss.jpg")