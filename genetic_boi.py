import random
from utils import (draw_rectangle, extract_features,
                   generate_random_rectangle, create_gif, save_images_to_folder, random_rectangle_set,
                   PICTURE_SIZE)
import numpy as np
from painting import Painting
import cv2
import time
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

MAX_SIZE = 20
EDGE_THICKNESS = 0
MULTIPLY_WEIGHTS = 50
POPULATION_SIZE = 20
RECTANGLE_AMOUNT = 350
ITERATION_LIMIT = 5000
K_TOURNAMENT = POPULATION_SIZE // 5
ELITE_PERCENT = 0.25
ELITE_NUMBER = int(np.ceil(ELITE_PERCENT * POPULATION_SIZE))
COLOR_ME_TENDERS = True # if false we choose color, else color randomly chosen
LOSS_TYPE = 'mse'

IMAGE_NAME = 'FELV-cat'
directory = f"./images_genetic/{IMAGE_NAME}/{POPULATION_SIZE}_{RECTANGLE_AMOUNT}_{ITERATION_LIMIT}_" \
            f"{MAX_SIZE}_{EDGE_THICKNESS}_{MULTIPLY_WEIGHTS}_{ELITE_PERCENT}_{LOSS_TYPE}_{COLOR_ME_TENDERS}"


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


def calculate_fitness_scores(population):
    fitness_scores = [-subject.loss for subject in population]
    return fitness_scores


def generate_distribution(fitness_scores):
    # Apply the softmax function
    # exp_scores = np.exp(fitness_scores)
    # probabilities = exp_scores / np.sum(exp_scores)
    scores = np.abs(fitness_scores)
    probabilities = scores / np.sum(scores)
    return probabilities


def mutation_1(child, og_image):
    # mutant_percentage = np.abs(np.random.normal(MUTANT_LOC, MUTANT_SCALE)) / 100
    child = np.array(child, dtype=object)
    genes_indices_to_be_mutated = np.random.choice(child.shape[0], size=1, replace=False)
    for i in genes_indices_to_be_mutated:
        child[i] = generate_random_rectangle(target_image=og_image, max_size=MAX_SIZE,
                                             edge_thickness=EDGE_THICKNESS, color_random=COLOR_ME_TENDERS)
    child = list(child)
    return child


def mutation_2(child, og_image):
    child.append(generate_random_rectangle(target_image=og_image, max_size=MAX_SIZE,
                                           edge_thickness=EDGE_THICKNESS, color_random=COLOR_ME_TENDERS))
    return child


def reproduce(mom, dad, target_image):
    mom_rectangles = np.array(mom.rectangle_list, dtype=object)
    dad_rectangles = np.array(dad.rectangle_list, dtype=object)

    n = max(mom_rectangles.size, dad_rectangles.size)

    def pad_array(arr, n):
        padding_length = n - len(arr)
        if padding_length > 0:
            return np.pad(arr, (0, padding_length), mode='constant', constant_values=None)
        return arr

    # Pad the arrays
    mom_padded = pad_array(mom_rectangles, n)
    dad_padded = pad_array(dad_rectangles, n)

    child_rectangles = []
    horizontal_cut = random.randint(0, 2)
    if horizontal_cut:
        horizontal_line = random.randint(0, PICTURE_SIZE[1])
        for i in range(n):
            if mom_padded[i] is not None and mom_padded[i].center[1] >= horizontal_line:
                child_rectangles.append(mom_padded[i])
            if dad_padded[i] is not None and dad_padded[i].center[1] < horizontal_line:
                child_rectangles.append(dad_padded[i])
    else:
        vertical_line = random.randint(0, PICTURE_SIZE[0])
        for i in range(n):
            if mom_padded[i] is not None and mom_padded[i].center[0] >= vertical_line:
                child_rectangles.append(mom_padded[i])
            if dad_padded[i] is not None and dad_padded[i].center[0] < vertical_line:
                child_rectangles.append(dad_padded[i])

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
    mutant_type = random.randint(1, 6)
    if mutant_type <= 1:
        child_rectangles = mutation_1(child_rectangles, target_image)
    elif mutant_type <= 4:
        child_rectangles = mutation_2(child_rectangles, target_image)
    return Painting(child_rectangles, target_image, mom.weights_matrix, LOSS_TYPE)


def tournament(pop, resized_img):
    contenders = np.random.choice(POPULATION_SIZE, size=K_TOURNAMENT, replace=False)
    best_subject = None
    # lowest_loss = np.inf
    # for contender_ind in range(K_TOURNAMENT):
    #     curr_loss = loss_function(pop[contenders[contender_ind]], resized_img)
    #     if curr_loss < lowest_loss:
    #         best_subject = pop[contenders[contender_ind]]
    #         lowest_loss = curr_loss

    return best_subject


def genetic_algorithm(target_img):
    # Generate random population
    curr_population = np.empty(shape=(POPULATION_SIZE,), dtype=object)
    for i in range(POPULATION_SIZE):
        rectangle_list = random_rectangle_set(number_of_rectangles=RECTANGLE_AMOUNT, target_image=target_img,
                                              max_size=MAX_SIZE, edge_thickness=EDGE_THICKNESS,
                                              color_random=COLOR_ME_TENDERS)
        curr_population[i] = Painting(rectangle_list, target_img, weights_matrix, LOSS_TYPE)

    losses = []
    for i in tqdm(range(ITERATION_LIMIT)):
        curr_population = np.sort(curr_population) # sort by loss
        fitness_scores = calculate_fitness_scores(curr_population)
        distribution = generate_distribution(fitness_scores)

        if i % 5 == 0:
            save_images_to_folder(directory + "/gif", [curr_population[0].image], target_img, [f"Generation: {i}"],
                                  [f"image_{i}"])

        losses.append(curr_population[0].loss)
        past_population = curr_population
        curr_population = np.empty(shape=(POPULATION_SIZE,), dtype=object)
        curr_population[:ELITE_NUMBER] = past_population[:ELITE_NUMBER]  # Keep the elite population

        for j in range(ELITE_NUMBER, POPULATION_SIZE):
            mom_and_dad = np.random.choice(len(distribution), size=2, p=distribution, replace=False)
            mom = past_population[mom_and_dad[0]]
            dad = past_population[mom_and_dad[1]]
            # mom = tournament(past_population, resized_image)
            # dad = tournament(past_population, resized_image)
            curr_population[j] = reproduce(mom, dad, target_img)

    return np.sort(curr_population), losses


if __name__ == '__main__':
    original_image_path = f'layouts/{IMAGE_NAME}.jpg'
    original_image = cv2.imread(original_image_path)
    target_img = cv2.resize(original_image, PICTURE_SIZE, interpolation=cv2.INTER_AREA)

    weights_matrix = extract_features(target_img, MULTIPLY_WEIGHTS)
    last_generation, losses = genetic_algorithm(target_img)
    losses.append(last_generation[0].loss)

    save_images_to_folder(directory + "/top5", [last_generation[i].image for i in range(5)], target_img,
                          [f"Generation: {ITERATION_LIMIT}"]*5)

    save_images_to_folder(directory + "/gif", [last_generation[0].image], target_img,
                          [f"Generation: {ITERATION_LIMIT}"], [f"image_{ITERATION_LIMIT}.png"])

    create_gif(directory + "/gif", directory + "/GIF.gif")

    plt.figure(figsize=(15, 7))  # Set the figure size
    plt.plot(losses, linestyle='-', color='blue')
    plt.title('Loss per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(directory + "/loss.jpg")