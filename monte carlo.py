import cv2
import numpy as np
import random
from utils import generate_random_rectangle, display_images_side_by_side, extract_features
import os
from tqdm import tqdm
from painting import Painting

MAX_SIZE = 20
EDGE_THICKNESS = 0
PICTURE_SIZE = (128, 128)
IMAGE_NAME = "FELV-cat"
RECTANGLE_AMOUNT = 500
NUM_SIMULATIONS = 1
NUM_SUCCESSORS = 5
MULTIPLY_WEIGHTS = 50
COLOR_ME_TENDERS = True
FEATURE_EXTRACT = True
LOSS_TYPE = 'delta'
directory = f"./images_monte/{IMAGE_NAME}/{RECTANGLE_AMOUNT}_{NUM_SIMULATIONS}_{MAX_SIZE}_{EDGE_THICKNESS}_" \
            f"{MULTIPLY_WEIGHTS}_{NUM_SUCCESSORS}_{COLOR_ME_TENDERS}_{LOSS_TYPE}"


class MCTS_Node(Painting):
    def __init__(self, rectangle_list, target_img, weights_matrix, loss_type=LOSS_TYPE,
                 feature_extract=FEATURE_EXTRACT, parent=None, rect=None, depth=0):
        super().__init__(rectangle_list, target_img, weights_matrix, loss_type, feature_extract)
        self.parent = parent
        self.children = []
        self.child_actions = []
        self.rect = rect
        self.wins = 0
        self.visits = 0
        self.untried_actions = self.generate_possible_actions(target_img)
        self.possible_actions = self.untried_actions.copy()
        self.depth = depth
        self.expanded = False

    @staticmethod
    def generate_possible_actions(target_img):
        # Generate potential rectangles
        actions = []
        for _ in range(NUM_SUCCESSORS):
            random_rect = generate_random_rectangle(target_image=target_img,
                                                    max_size=MAX_SIZE,
                                                    edge_thickness=EDGE_THICKNESS,
                                                    color_random=COLOR_ME_TENDERS)
            while random_rect in actions:
                print("heyo")
                random_rect = generate_random_rectangle(target_image=target_img,
                                                        max_size=MAX_SIZE,
                                                        edge_thickness=EDGE_THICKNESS,
                                                        color_random=COLOR_ME_TENDERS)
            actions.append(random_rect)
        return actions

    def uct_select_child(self, c_param=0.2):
        # Select child with highest UCT score
        uct_weights = [(child.wins / child.visits) +
                       c_param * np.sqrt((2 * np.log(self.visits) / child.visits)) for child in
                       self.children if child.expanded]
        return self.children[np.argmax(uct_weights)]

    def add_child(self, move):
        new_rect_list = self.rectangle_list.copy()
        new_rect_list.append(move)
        child_node = MCTS_Node(new_rect_list, self.target_image, self.weights_matrix, parent=self, rect=move,
                               depth=self.depth + 1)
        self.children.append(child_node)
        self.child_actions.append(move)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self):
        action = random.choice(self.untried_actions)
        child_actions = [child.rect for child in self.children]
        if action in child_actions:
            child = self.children[child_actions.index(action)]
        else:
            child = self.add_child(action)
        self.untried_actions.remove(action)
        child.expanded = True
        return child

    def randomly_choose_child(self):
        action = random.choice(self.possible_actions)
        child_actions = [child.rect for child in self.children]
        if action in child_actions:
            return self.children[child_actions.index(action)]
        else:
            return self.add_child(action)

    def tree_policy(self):
        current_node = self
        while current_node.depth < RECTANGLE_AMOUNT:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.uct_select_child(c_param=0.1)
        return current_node

    def backpropagate(self, result):
        self.update(result)
        if self.parent:
            self.parent.backpropagate(result)

    def best_action(self):
        for i in range(NUM_SIMULATIONS):
            v = self.tree_policy()
            rollout_img = v.rollout(steps=RECTANGLE_AMOUNT-self.depth)
            reward = -rollout_img.loss
            v.backpropagate(reward)
        return self.uct_select_child(c_param=0.)

    def rollout(self, steps):
        curr_state = self
        for _ in range(steps):
            curr_state = curr_state.randomly_choose_child()
        return curr_state


def main(target_img, weights_matrix):
    curr_node = MCTS_Node([], target_img, weights_matrix)
    for _ in tqdm(range(RECTANGLE_AMOUNT)):
        curr_node = curr_node.best_action()
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

