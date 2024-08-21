from utils import display_images_side_by_side, PICTURE_SIZE
import cv2

MAX_SIZE = 20
EDGE_THICKNESS = 0
MULTIPLY_WEIGHTS = 1
POPULATION_SIZE = 20
RECTANGLE_AMOUNT = 350
ITERATION_LIMIT = 5000
MUTANT_LOC = 5
MUTANT_SCALE = 2

def main():
    #load original picture
    original_image_path = 'layouts/image.jpg'
