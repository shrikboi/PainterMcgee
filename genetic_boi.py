import os
import random
from rectangle import Rectangle
from utils import draw_rectangle, display_images_side_by_side
import numpy as np
import cv2
import time
import heapq
from tqdm import tqdm

MAX_WIDTH = 30
MAX_HEIGHT = 30
POPULATION_SIZE = 100
RECTANGLE_AMOUNT = 1000
ITERATION_LIMIT = 500
PICTURE_SIZE = (128, 128)
MUTANT_LOC = 20
MUTANT_SCALE = 1


def generate_random_rectangle(edge_thickness=0):
    width = random.randint(1, MAX_WIDTH)  # Random width between 1 and 100
    height = random.randint(1, MAX_HEIGHT)  # Random height between 1 and 100
    degree = random.randint(0, 360)  # Random degree between 0 and 360
    center_width = random.randint(0, PICTURE_SIZE[0])
    center_height = random.randint(0, PICTURE_SIZE[1])
    color_r = random.randint(0, 255)
    color_g = random.randint(0, 255)
    color_b = random.randint(0, 255)

    if edge_thickness:
        edge_thickness = random.randint(1, edge_thickness)  # Random edge_thickness between 1 and 10
    return (Rectangle((width, height), degree, edge_thickness),
            (center_width, center_height), (color_r, color_g, color_b))


def loss_function(subject, og_image):
    current_painting = np.ones((*PICTURE_SIZE[::-1], 3), dtype=np.uint8) * 255
    for rectangle, center, color in subject:
        color = calculate_color(rectangle, center, og_image)
        draw_rectangle(current_painting, center, rectangle, color)

    squared_diff = (og_image - current_painting) ** 2
    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)
    return mean_squared_diff


def generate_distribution(population, og_image):
    fitness_scores = np.zeros(POPULATION_SIZE)
    for j, subject in enumerate(population):
        fitness_scores[j] = loss_function(subject, og_image)

    fitness_scores /= fitness_scores.sum()
    fitness_scores = 1 - fitness_scores  # Fitness scores was a loss function and not scores
    fitness_scores /= fitness_scores.sum()
    return fitness_scores


def calculate_color(rectangle, center, og_image):
    rect = (center, rectangle.size, rectangle.angle)

    # Calculate the bounding box of the rotated rectangle in original coordinates
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # Get the coordinates of the bounding box
    x, y, w, h = cv2.boundingRect(box)

    # Ensure the bounding box is within the image boundaries
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, og_image.shape[1] - x)
    h = min(h, og_image.shape[0] - y)

    # Extract the bounding box region from the original image
    bounding_box_roi = og_image[y:y + h, x:x + w]

    # Calculate the center of the bounding box region
    bounding_box_center = (bounding_box_roi.shape[1] // 2, bounding_box_roi.shape[0] // 2)

    # Get the rotation matrix for the bounding box region
    M = cv2.getRotationMatrix2D(bounding_box_center, rectangle.angle, 1.0)

    # Rotate the bounding box region
    rotated_roi = cv2.warpAffine(bounding_box_roi, M, (bounding_box_roi.shape[1], bounding_box_roi.shape[0]))

    # Calculate the new center of the rotated ROI
    new_center = (rotated_roi.shape[1] // 2, rotated_roi.shape[0] // 2)

    # Calculate the bounding box of the rotated rectangle in the rotated ROI coordinates
    rect = (new_center, rectangle.size, 0)  # Angle is 0 because we already rotated the region
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    x, y, w, h = cv2.boundingRect(box)

    # Ensure the ROI is within the rotated region boundaries
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, rotated_roi.shape[1] - x)
    h = min(h, rotated_roi.shape[0] - y)

    # Extract the final ROI from the rotated bounding box region
    final_roi = rotated_roi[y:y + h, x:x + w]

    # Calculate the average color
    avg_color = cv2.mean(final_roi)[:3]  # Exclude the alpha channel if present
    return avg_color

def rectangle_scores(rectangles, og_image):
    scores = np.zeros(rectangles.shape[0])

    for j, (rectangle, center, color) in enumerate(rectangles):
        avg_color = calculate_color(rectangle, center, og_image)
        scores[j] = np.mean((np.array(avg_color) - np.array(color))**2)

    scores = scores/scores.sum()
    scores = 1 - scores
    scores = scores/scores.sum()
    return scores


def reproduce(mom, dad, og_image):
    mom = np.array(mom, dtype=object)
    dad = np.array(dad, dtype=object)

    mom_rectangle_scores = rectangle_scores(mom, og_image)
    dad_rectangle_scores = rectangle_scores(dad, og_image)

    mom_genes = np.random.choice(mom.shape[0], size=int(RECTANGLE_AMOUNT / 2), replace=False, p=mom_rectangle_scores)
    dad_genes = np.random.choice(dad.shape[0], size=int(RECTANGLE_AMOUNT / 2), replace=False, p=dad_rectangle_scores)

    child_genes = np.concatenate((mom_genes, dad_genes), axis=0)

    # Mutant me up scotty
    mutant_percentage = np.abs(np.random.normal(MUTANT_LOC, MUTANT_SCALE)) / 100

    genes_indices_to_be_mutated = (np.random.choice(child_genes.shape[0],
                                                    size=int(np.ceil(RECTANGLE_AMOUNT*mutant_percentage)),
                                                    replace=False))
    for mutation in genes_indices_to_be_mutated:
        child_genes[mutation] = generate_random_rectangle()

    return list(child_genes)


if __name__ == '__main__':
    print(f"The parameters are:")
    print(f"Population size {POPULATION_SIZE}, {RECTANGLE_AMOUNT} rectangles,"
          f" {ITERATION_LIMIT} iterations, mutant loc {MUTANT_LOC}, mutant scale {MUTANT_SCALE},"
          f" max width {MAX_WIDTH}")
    original_image_path = 'layouts/FELV-cat.jpg'
    original_image = cv2.imread(original_image_path)
    resized_image = cv2.resize(original_image, PICTURE_SIZE, interpolation=cv2.INTER_AREA)

    # Generate random population
    curr_population = []
    for _ in range(POPULATION_SIZE):
        curr_population.append([generate_random_rectangle() for _ in range(RECTANGLE_AMOUNT)])

    for i in tqdm(range(ITERATION_LIMIT)):
        start_iter_time = time.time()
        # print(f"Iteration {i}")
        past_population = curr_population

        curr_population = []
        distribution = generate_distribution(past_population, resized_image)
        for _ in range(POPULATION_SIZE):
            mom_and_dad = np.random.choice(len(distribution), size=2, p=distribution, replace=False)
            mom = past_population[mom_and_dad[0]]
            dad = past_population[mom_and_dad[1]]
            curr_population.append(reproduce(mom, dad, resized_image))

        # print(f"Time taken: {time.time() - start_iter_time}")

    # best_subject = None
    # lowest_loss = np.inf
    # for i in range(POPULATION_SIZE):
    #     curr_loss = loss_function(curr_population[i], resized_image)
    #     if curr_loss < lowest_loss:
    #         best_subject = curr_population[i]
    #         lowest_loss = curr_loss



    # Initialize a list to keep the top 5 subjects with their losses
    top_5 = []
    for i in range(POPULATION_SIZE):
        curr_loss = loss_function(curr_population[i], resized_image)

        # If we have fewer than 5 subjects, simply add the current one
        if len(top_5) < 5:
            heapq.heappush(top_5, (curr_loss, curr_population[i]))
        else:
            # If the current loss is lower than the highest loss in the top 5, replace it
            if curr_loss < top_5[0][0]:
                heapq.heappushpop(top_5, (curr_loss, curr_population[i]))

    # Extract the best 5 subjects from the heap
    best_subjects = [subject for loss, subject in heapq.nsmallest(5, top_5)]

    best_paintings = []
    directory = f"./images_genetic/{POPULATION_SIZE}_{RECTANGLE_AMOUNT}_{ITERATION_LIMIT}_{MUTANT_LOC}_{MUTANT_SCALE}_{MAX_WIDTH}_"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, subject in enumerate(best_subjects):
        best_paintings.append(np.ones((*PICTURE_SIZE[::-1], 3), dtype=np.uint8) * 255)
        for rectangle, center, color in subject:
            color = calculate_color(rectangle, center, resized_image)
            draw_rectangle(best_paintings[i], center, rectangle, color)

        cv2.imwrite(
            os.path.join(directory, f"image_{i}.png"),
            display_images_side_by_side(resized_image, best_paintings[i]))

