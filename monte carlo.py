import cv2
import numpy as np
import random
from utils import generate_random_rectangle, draw_rectangle, display_images_side_by_side
from math import sqrt, log
import os

NUMBER_OF_RECTANGLES = 400
MAX_REC_WIDTH = 30
MAX_REC_HEIGHT = 30
PICTURE_SIZE = (128, 128)
ORIGINAL_PICTURE_NAME = "FELV-cat.jpg"

class Node:
    def __init__(self, image, target_img, parent=None, rect=None, depth=0):
        self.image = image
        self.parent = parent
        self.children = []
        self.rect = rect
        self.wins = 0
        self.visits = 0
        self.untried_actions = self.generate_possible_actions(target_img)
        self.depth = depth


    @staticmethod
    def generate_possible_actions(target_img):
        # Generate potential rectangle properties
        actions = []
        for _ in range(5):  # Limit the number of potential actions for simplicity
            rectangle = generate_random_rectangle(og_image=target_img)
            actions.append(rectangle)
        return actions

    def uct_select_child(self):
        # Select child with highest UCT score
        t = sum(child.visits for child in self.children)
        log_t = log(t) if t > 0 else 0
        return max(self.children, key=lambda child: child.wins / child.visits + sqrt(2 * log_t / child.visits) if t > 0 else 0)

    def add_child(self, rect, target_img):
        new_img = draw_rectangle(self.image.copy(), rect)
        child_node = Node(new_img, target_img, parent=self, rect=rect, depth=self.depth + 1)
        self.untried_actions.remove(rect)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result


def rollout(image, target_img, steps=3):
    for _ in range(steps):
        rect = random.choice(Node.generate_possible_actions(target_img))
        image = draw_rectangle(image.copy(), rect)
    return image


def evaluate(image, target):
    return -cv2.norm(image, target, cv2.NORM_L2)  # Use negative L2 norm for 'wins'


def mcts(root, target_img, max_depth=400, iterations=100):
    best_node = None
    best_score = float('-inf')

    for _ in range(iterations):
        node = root
        # Selection and Expansion
        while node.children or node.untried_actions:
            if node.untried_actions:
                rect = random.choice(node.untried_actions)
                node = node.add_child(rect, target_img)
            else:
                node = node.uct_select_child()

            if node.depth == max_depth:
                score = evaluate(node.image, target_img)
                if score > best_score:
                    best_score = score
                    best_node = node
                break

        # Rollout and Backpropagation
        if node.depth < max_depth:
            rollout_img = rollout(node.image.copy(), target_img)
            result = evaluate(rollout_img, target_img)
            while node is not None:
                node.update(result)
                node = node.parent

    return best_node.image if best_node else root.image


# Initialize
target_img = cv2.imread('./layouts/' + ORIGINAL_PICTURE_NAME)
target_img = cv2.resize(target_img, (128, 128))
root_node = Node(np.ones_like(target_img)*255, target_img)


best_approximation = mcts(root_node, target_img, max_depth=500, iterations=1000)
directory = f"./images_monte/"
if not os.path.exists(directory):
    os.makedirs(directory)
cv2.imwrite(
    os.path.join(directory, ORIGINAL_PICTURE_NAME),
    display_images_side_by_side(target_img, best_approximation))

