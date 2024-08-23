import cv2
import numpy as np
import random
from utils import generate_random_rectangle, draw_rectangle, display_images_side_by_side, extract_features, \
    random_rectangle_set
from math import sqrt, log
import os
import colour
from tqdm import tqdm

MAX_SIZE = 20
EDGE_THICKNESS = 0
PICTURE_SIZE = (128, 128)
IMAGE_NAME = "FELV-cat"
RECTANGLE_AMOUNT = 500
NUM_SIMULATIONS = 100
NUM_SUCCESSORS = 5
MULTIPLY_WEIGHTS = 50
COLOR_ME_TENDERS = True
AMOUNT_OF_MOVES = 5000
LOSS_TYPE = 'delta'
directory = f"./images_monte/{IMAGE_NAME}/{RECTANGLE_AMOUNT}_{NUM_SIMULATIONS}_{MAX_SIZE}_{EDGE_THICKNESS}_" \
            f"{MULTIPLY_WEIGHTS}_{AMOUNT_OF_MOVES}_{NUM_SUCCESSORS}_{COLOR_ME_TENDERS}_{LOSS_TYPE}"


class Node:
    def __init__(self, image, target_img, weights_matrix, parent=None, rect=None, depth=0):
        self.image = image
        self.parent = parent
        self.children = []
        self.rect = rect
        self.wins = 0
        self.visits = 0
        self.untried_actions = self.generate_possible_actions(target_img)
        self.depth = depth
        self.weights_matrix = weights_matrix

    @staticmethod
    def generate_possible_actions(target_img):
        # Generate potential rectangle properties
        actions = random_rectangle_set(NUM_SUCCESSORS, target_img, MAX_SIZE, EDGE_THICKNESS, COLOR_ME_TENDERS)
        return actions

    def uct_select_child(self, c_param=0.2):
        # Select child with highest UCT score
        uct_weights = [(child.wins / child.visits) +
                           c_param * np.sqrt((2 * np.log(self.visits) / child.visits)) for child in self.children]
        return self.children[np.argmax(uct_weights)]

    def add_child(self, move, target_img):
        new_img = draw_rectangle(self.image.copy(), move)
        child_node = Node(new_img, target_img, self.weights_matrix, parent=self, rect=move, depth=self.depth + 1)
        self.untried_actions.remove(move)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self, target_img):
        action = random.choice(self.untried_actions)
        return self.add_child(action, target_img)

    def tree_policy(self, target_img):
        current_node = self
        while current_node.depth < RECTANGLE_AMOUNT:
            if not current_node.is_fully_expanded():
                return current_node.expand(target_img)
            else:
                current_node = current_node.uct_select_child(c_param=0.1)
        return current_node

    def backpropagate(self, result):
        self.update(result)
        if self.parent:
            self.parent.backpropagate(result)

    def best_action(self, target_img):
        for i in range(NUM_SIMULATIONS):
            v = self.tree_policy(target_img)
            rollout_img = v.rollout(target_img, steps=RECTANGLE_AMOUNT-self.depth)
            if LOSS_TYPE == 'mse':
                reward = self.mse_loss(target_img)
            elif LOSS_TYPE == "delta":
                reward = self.delta_e_loss(target_img)
            v.backpropagate(reward)

        return self.uct_select_child(c_param=0.)

    def rollout(self, target_img, steps):
        # current_rollout_state = self
        image = self.image.copy()
        for _ in range(steps):
            rect = random.choice(Node.generate_possible_actions(target_img))
            image = draw_rectangle(image.copy(), rect)
        return image


def mse_loss(image, target_img, weights_matrix):
    weights_stacked = np.stack([weights_matrix, weights_matrix, weights_matrix], axis=-1)
    return -np.mean(
        np.square(image.astype(np.float32) - target_img.astype(np.float32)) * weights_stacked)

def delta_e_loss(image, target_img, weights_matrix):
    image = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    target = cv2.cvtColor(target_img.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)

    delta_e = colour.difference.delta_e.delta_E_CIE2000(image, target)
    return -np.mean(delta_e * weights_matrix)


def main(target_img, weights_matrix):
    curr_node = Node(np.ones_like(target_img)*255, target_img, weights_matrix)
    for _ in tqdm(range(RECTANGLE_AMOUNT)):
        curr_node = curr_node.best_action(target_img)
    return curr_node


# Initialize
target_img = cv2.imread('./layouts/' + IMAGE_NAME+".jpg")
target_img = cv2.resize(target_img, (128, 128))
weights_matrix = extract_features(target_img, MULTIPLY_WEIGHTS)
best_approximation = main(target_img, weights_matrix)
if not os.path.exists(directory):
    os.makedirs(directory)
cv2.imwrite(
    os.path.join(directory, IMAGE_NAME+".jpg"),
    display_images_side_by_side(target_img, best_approximation.image, None))

