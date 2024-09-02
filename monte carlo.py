import cv2
import numpy as np
import random
from utils import (generate_random_rectangle, display_images_side_by_side, extract_features, PICTURE_SIZE,
                   save_images_to_folder)
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from painting import BeamSearch_Node, LOSS

RECTANGLE_AMOUNT = 500 # Number of rectangles in each painting
NUM_SIMULATIONS = 100 # Number of simulations to perform at each step
NUM_SUCCESSORS = 10 # Number of possible actions (rectangles) to generate per node
LOSS_TYPE = LOSS.EUCLIDEAN # Type of loss function to use
IMAGE_NAME = 'FELV-cat' # Name of the target image

MULTIPLY_WEIGHTS = 50
COLOR_ME_TENDERS = True # if true color randomly chosen, else color is chosen by average color in target rectangle area
EDGE_THICKNESS = 0 # Thickness of the rectangle edges
MAX_SIZE = 20 # Maximum size of rectangles

# Directory to save generated images
directory = f"./images_monte/{IMAGE_NAME}/{RECTANGLE_AMOUNT}_{NUM_SIMULATIONS}_{MAX_SIZE}_{EDGE_THICKNESS}_" \
            f"{MULTIPLY_WEIGHTS}_{NUM_SUCCESSORS}_{COLOR_ME_TENDERS}_{LOSS_TYPE}"


class MCTS_Node(BeamSearch_Node):
    """
    A class representing a node in the Monte Carlo Tree Search (MCTS) algorithm.
    Inherits from BeamSearch_Node and extends it with MCTS-specific functionality.
    """
    def __init__(self, rectangle_list, target_img, weights_matrix, loss_type=LOSS_TYPE,
                 parent=None, rect=None, depth=0):
        super().__init__(rectangle_list, target_img, weights_matrix, loss_type, parent=parent)
        self.children = []
        self.child_actions = []
        self.rect = rect
        self.wins = 0
        self.visits = 0
        self.untried_actions = self.generate_possible_actions(target_img)
        self.possible_actions = self.untried_actions.copy()
        self.depth = depth

    @staticmethod
    def generate_possible_actions(target_img):
        """
        Generate a list of possible rectangle actions that can be taken from this node.

        @param target_img: The target image we want to approximate.
        @return: List of possible rectangle actions.
        """
        actions = []
        for _ in range(NUM_SUCCESSORS):
            random_rect = generate_random_rectangle(target_img=target_img,
                                                    max_size=MAX_SIZE,
                                                    edge_thickness=EDGE_THICKNESS,
                                                    color_random=COLOR_ME_TENDERS)
            while random_rect in actions:
                random_rect = generate_random_rectangle(target_img=target_img,
                                                        max_size=MAX_SIZE,
                                                        edge_thickness=EDGE_THICKNESS,
                                                        color_random=COLOR_ME_TENDERS)
            actions.append(random_rect)
        return actions

    def uct_select_child(self, c_param=0.2):
        """
        Select the child node with the highest Upper Confidence Bound applied to Trees (UCT) score.
        @param c_param: The exploration parameter (default is 0.2).
        @return: The child node with the highest UCT score.
        """
        # Select child with highest UCT score
        uct_weights = [(child.wins / child.visits) +
                       c_param * np.sqrt((2 * np.log(self.visits) / child.visits)) for child in
                       self.children]
        return self.children[np.argmax(uct_weights)]

    def add_child(self, move):
        """
        Add a new child node resulting from a given move (rectangle action).
        @param move: The rectangle action that leads to the new child node.
        @return: The newly created child node.
        """
        new_rect_list = self.rectangle_list.copy()
        new_rect_list.append(move)
        child_node = MCTS_Node(new_rect_list, self.target_image, self.weights_matrix, parent=self, rect=move,
                               depth=self.depth + 1)
        self.children.append(child_node)
        self.child_actions.append(move)
        return child_node

    def update(self, result):
        """
        Update the node's statistics (wins and visits) based on the result of a simulation.
        @param result: The result of the simulation (reward).
        @return:
        """
        self.visits += 1
        self.wins += result

    def is_fully_expanded(self):
        """
        Check if all possible actions from this node have been tried.
        @return: True if all actions have been tried, False otherwise.
        """
        return len(self.untried_actions) == 0

    def expand(self):
        """
        Expand the node by trying an untried action and adding a corresponding child node.
        @return: The newly created child node.
        """
        action = random.choice(self.untried_actions)
        child = self.add_child(action)
        self.untried_actions.remove(action)
        return child

    def randomly_choose_child(self):
        """
        Randomly select a child node from the possible actions.
        @return: The selected child node.
        """
        action = random.choice(self.possible_actions)
        child_actions = [child.rect for child in self.children]
        if action in child_actions:
            return self.children[child_actions.index(action)]
        else:
            return self.add_child(action)

    def tree_policy(self):
        """
        Navigate the tree according to the tree policy, expanding nodes as needed.
        @return: The node selected by the tree policy.
        """
        current_node = self
        while current_node.depth < RECTANGLE_AMOUNT:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.uct_select_child(c_param=0.1)
        return current_node

    def backpropagate(self, result):
        """
        Backpropagate the result of a simulation up the tree, updating nodes along the way.
        @param result: The result of the simulation (reward).
        @return:
        """
        self.update(result)
        if self.parent:
            self.parent.backpropagate(result)

    def best_action(self):
        """
        Determine the best action to take from this node using MCTS.
        @return: The child node representing the best action.
        """
        for i in range(NUM_SIMULATIONS):
            v = self.tree_policy()
            rollout_img = v.rollout(steps=RECTANGLE_AMOUNT-self.depth)
            reward = -rollout_img.loss
            v.backpropagate(reward)
        return self.uct_select_child(c_param=0.)

    def rollout(self, steps):
        """
        Perform a rollout (simulation) from this node by randomly generating rectangles.
        @param steps: The number of steps to simulate.
        @return: A new MCTS_Node representing the end state of the rollout.
        """
        rectangle_list = self.rectangle_list.copy()
        for _ in range(steps):
            rect = generate_random_rectangle(self.target_image, MAX_SIZE)
            rectangle_list.append(rect)
        return MCTS_Node(rectangle_list, self.target_image, self.weights_matrix)


def monte_carlo_tree_search(target_img, weights_matrix):
    """
    Perform Monte Carlo Tree Search (MCTS) to find the best sequence of rectangles that approximate the target image.
    @param target_img: The target image to be approximated.
    @param weights_matrix: The weight matrix extracted from the target image.
    @return: The final MCTS_Node representing the best sequence found.
    """
    curr_node = MCTS_Node([], target_img, weights_matrix)
    for _ in tqdm(range(RECTANGLE_AMOUNT)):
        curr_node = curr_node.best_action()
    return curr_node


if __name__ == '__main__':
    target_img = cv2.imread('./layouts/' + IMAGE_NAME+".jpg")
    target_img = cv2.resize(target_img, PICTURE_SIZE)

    weights_matrix = extract_features(target_img, MULTIPLY_WEIGHTS)
    best_approximation = monte_carlo_tree_search(target_img, weights_matrix)

    path = best_approximation.get_path()
    losses = [node.loss for node in path]

    save_images_to_folder(directory, [best_approximation.image], target_img, titles=[f"final_image.png"])
    plt.figure(figsize=(8, 7))  # Set the figure size
    plt.plot(losses, linestyle='-', color='blue')
    plt.title('Loss per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(directory + "/loss.jpg")

