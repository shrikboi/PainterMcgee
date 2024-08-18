import os
import random
from utils import (draw_rectangle, display_images_side_by_side, extract_features, calculate_color,
                   generate_random_rectangle, display_image, create_gif, save_paintings_to_folder)
import numpy as np
import cv2
import time
import heapq
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

MAX_REC_WIDTH = 30
MAX_REC_HEIGHT = 30
EDGE_THICKNESS = 1
MULTIPLY_WEIGHTS = 50
POPULATION_SIZE = 20
RECTANGLE_AMOUNT = 200
ITERATION_LIMIT = 5000
PICTURE_SIZE = (128, 128)
MUTANT_LOC = 5
MUTANT_SCALE = 2
K_TOURNAMENT = POPULATION_SIZE // 5
directory = f"./images_genetic/{POPULATION_SIZE}_{RECTANGLE_AMOUNT}_{ITERATION_LIMIT}_{MUTANT_LOC}_" \
                f"{MUTANT_SCALE}_{MAX_REC_WIDTH}_{EDGE_THICKNESS}_{MULTIPLY_WEIGHTS}"

# if false we choose color, else color randomly chosen
COLOR_ME_TENDERS = False

# Load a pre-trained VGG16 model
vgg = models.vgg16(weights="VGG16_Weights.DEFAULT").features


def no_extract_features(model, x):
    layers = [3, 8, 15, 22]  # Select layers at different depths
    features = []
    for i, layer in enumerate(model):
        x = layer(x)
        if i in layers:
            features.append(x)
    return features


def calculate_perceptual_loss(rectangles_list, og_image):
    start_perceptual = time.time()

    curr_painting = np.ones((*PICTURE_SIZE[::-1], 3), dtype=np.uint8) * 255
    for rectangle in rectangles_list:
        draw_rectangle(curr_painting, rectangle)

    img1 = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(curr_painting, cv2.COLOR_BGR2RGB)

    # Define transformations: resize, convert to tensor, normalize
    start_preprocess = time.time()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize images to 224x224 as VGG expects
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img1 = preprocess(img1).unsqueeze(0)  # Add batch dimension
    img2 = preprocess(img2).unsqueeze(0)  # Add batch dimension
    # print("preprocess time: ", time.time() - start_preprocess)

    # Ensure the model is in evaluation mode
    vgg.eval()

    # Extract features

    start_extract = time.time()
    features_img1 = no_extract_features(vgg, img1)
    features_img2 = no_extract_features(vgg, img2)
    # print("extract time: ", time.time() - start_extract)

    start_loss = time.time()
    perceptual_loss = 0.0
    for f1, f2 in zip(features_img1, features_img2):
        perceptual_loss += torch.nn.functional.mse_loss(f1, f2)
    # print("loss time: ", time.time() - start_loss)
    # print("perceptual loss total time: ", time.time() - start_perceptual)
    return perceptual_loss


def loss_function(subject, og_image, weights_matrix):
    l2_start_time = time.time()
    current_painting = np.ones((*PICTURE_SIZE[::-1], 3), dtype=np.uint8) * 255
    for rectangle in subject:
        draw_rectangle(current_painting, rectangle)

    squared_diff = np.square(og_image - current_painting)
    weights_stacked = np.stack([weights_matrix, weights_matrix, weights_matrix], axis=-1)
    squared_diff = squared_diff*weights_stacked
    # Calculate the mean of the squared differences
    MSE = np.mean(squared_diff)
    RMSE = np.sqrt(MSE)
    # print("l2 time is ", time.time()-l2_start_time)
    return RMSE


def calculate_fitness_scores(population, og_image, weights_matrix):
    fitness_scores = np.zeros(len(population))
    for j, subject in enumerate(population):
        fitness_scores[j] = loss_function(subject, og_image, weights_matrix)
    return -fitness_scores


def generate_distribution(fitness_scores):
    # Apply the softmax function
    exp_scores = np.exp(fitness_scores)
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities


def rectangle_scores(rectangles, og_image):
    scores = np.zeros(rectangles.shape[0])

    for j, rectangle in enumerate(rectangles):
        avg_color = calculate_color(rectangle.size, rectangle.center, rectangle.angle, og_image)
        scores[j] = np.mean((np.array(avg_color) - np.array(rectangle.color)) ** 2)

    # Negate the loss scores
    negated_scores = -scores
    # Apply the softmax function
    exp_scores = np.exp(negated_scores)
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities


def reproduce(mom, dad, og_image):
    mom = np.array(mom, dtype=object)
    dad = np.array(dad, dtype=object)

    child = []
    horizontal_cut = random.randint(0, 2)
    if horizontal_cut:
        horizontal_line = random.randint(0, PICTURE_SIZE[1])
        for rectangle in mom:
            if rectangle.center[1] >= horizontal_line:
                child.append(rectangle)
        for rectangle in dad:
            if rectangle.center[1] < horizontal_line:
                child.append(rectangle)
    else:
        vertical_line = random.randint(0, PICTURE_SIZE[0])
        for rectangle in mom:
            if rectangle.center[0] >= vertical_line:
                child.append(rectangle)
        for rectangle in dad:
            if rectangle.center[0] < vertical_line:
                child.append(rectangle)

    # best_child = None
    # lowest_loss = np.inf
    # for contender_ind in range(K_TOURNAMENT):
    #     mom_genes = np.random.choice(mom.shape[0], size=int(RECTANGLE_AMOUNT / 2), replace=False)
    #     dad_genes = np.random.choice(dad.shape[0], size=int(RECTANGLE_AMOUNT / 2), replace=False)
    #
    #     child = np.concatenate((mom[mom_genes], dad[dad_genes]), axis=0)
    #     curr_loss = loss_function(child, og_image)
    #     if curr_loss < lowest_loss:
    #         best_child = child
    #         lowest_loss = curr_loss

    # Mutant me up scotty
    mutant_percentage = np.abs(np.random.normal(MUTANT_LOC, MUTANT_SCALE)) / 100

    child = np.array(child, dtype=object)
    genes_indices_to_be_mutated = (np.random.choice(child.shape[0],
                                                    size=int(np.ceil(RECTANGLE_AMOUNT * mutant_percentage)),
                                                    replace=False))
    for mutation in genes_indices_to_be_mutated:
        child[mutation] = generate_random_rectangle(og_image=og_image,
                                                    max_width=MAX_REC_WIDTH,
                                                    max_height=MAX_REC_HEIGHT,
                                                    edge_thickness=EDGE_THICKNESS)

    return list(child)


def tournament(pop, resized_img):
    contenders = np.random.choice(POPULATION_SIZE, size=K_TOURNAMENT, replace=False)
    best_subject = None
    lowest_loss = np.inf
    for contender_ind in range(K_TOURNAMENT):
        curr_loss = loss_function(pop[contenders[contender_ind]], resized_img)
        if curr_loss < lowest_loss:
            best_subject = pop[contenders[contender_ind]]
            lowest_loss = curr_loss

    return best_subject


def add_gaussian_noise(rgb_tuple, mean=0, stddev=15):
    """
    Adds Gaussian noise to each component of the RGB tuple.

    :param rgb_tuple: A tuple of 3 floats representing an RGB color.
    :param mean: Mean of the Gaussian noise.
    :param stddev: Standard deviation of the Gaussian noise.
    :return: A tuple of 3 floats with added Gaussian noise, clamped between 0 and 1.
    """
    noisy_rgb = []
    for value in rgb_tuple:
        noise = np.random.normal(mean, stddev)
        noisy_value = value + noise
        # Clamp the value between 0 and 1
        noisy_value = min(max(noisy_value, 0), 255)
        noisy_rgb.append(noisy_value)

    return tuple(noisy_rgb)


def top_indices(arr, percent):
    # Calculate the number of top elements corresponding to percent%
    n_top = int(np.ceil(len(arr) * percent))

    # Get the indices of the sorted array in descending order
    sorted_indices = np.argsort(arr)[::-1]

    # Return the indices of the top 20% elements
    return sorted_indices[:n_top]





if __name__ == '__main__':
    print(f"The parameters are:")
    print(f"Population size {POPULATION_SIZE}, {RECTANGLE_AMOUNT} rectangles,"
          f" {ITERATION_LIMIT} iterations, mutant loc {MUTANT_LOC}, mutant scale {MUTANT_SCALE},"
          f" max width {MAX_REC_WIDTH}")
    original_image_path = 'layouts/FELV-cat.jpg'
    original_image = cv2.imread(original_image_path)
    resized_image = cv2.resize(original_image, PICTURE_SIZE, interpolation=cv2.INTER_AREA)

    weights_matrix = extract_features(resized_image, MULTIPLY_WEIGHTS)

    # Generate random population
    curr_population = np.empty(shape=(POPULATION_SIZE,), dtype=list)
    for i in range(POPULATION_SIZE):
        curr_population[i] = [generate_random_rectangle(og_image=resized_image, max_height=MAX_REC_HEIGHT,
                                                        max_width=MAX_REC_WIDTH, edge_thickness=EDGE_THICKNESS)
                              for _ in range(RECTANGLE_AMOUNT)]

    best_of_every_round = []
    for i in tqdm(range(ITERATION_LIMIT)):
        start_iter_time = time.time()

        past_population = curr_population
        curr_population = np.empty(shape=(POPULATION_SIZE,), dtype=list)

        fitness_scores = calculate_fitness_scores(past_population, resized_image, weights_matrix)
        if i % 5 == 0:
            top_5 = top_indices(fitness_scores, 5/POPULATION_SIZE)
            best_of_every_round.append(past_population[top_5[0]])
            print("top 5 avg fitness score is ", np.average(fitness_scores[top_5]))


        indices = top_indices(fitness_scores, 0.25)
        curr_population[:len(indices)] = past_population[indices]

        distribution = generate_distribution(fitness_scores)
        for j in range(len(indices), POPULATION_SIZE):
            mom_and_dad = np.random.choice(len(distribution), size=2, p=distribution, replace=False)
            mom = past_population[mom_and_dad[0]]
            dad = past_population[mom_and_dad[1]]
            # mom = tournament(past_population, resized_image)
            # dad = tournament(past_population, resized_image)

            curr_population[j] = reproduce(mom, dad, resized_image)

        # print(f"Time taken: {time.time() - start_iter_time}")

    fitness_scores = calculate_fitness_scores(curr_population, resized_image, weights_matrix)
    best_of_every_round.append(curr_population[top_indices(fitness_scores, 1 / POPULATION_SIZE)[0]])
    top_5 = top_indices(fitness_scores, 5/POPULATION_SIZE)

    save_paintings_to_folder(directory+"/top5", curr_population[top_5], resized_image)
    save_paintings_to_folder(directory+"/gif", best_of_every_round, resized_image)

    create_gif(directory+"/gif", directory+"/GIF.gif")