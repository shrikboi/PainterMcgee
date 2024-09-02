import random
from utils import (extract_features,
                   generate_random_rectangle, create_gif, save_images_to_folder,
                   random_rectangle_set, PICTURE_SIZE)
import numpy as np
from painting import Painting, LOSS
import cv2
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Constants defining the genetic algorithm's parameters
POPULATION_SIZE = 20 # Number of individuals in the population
RECTANGLE_AMOUNT = 350 # Number of starting rectangles
MUTATION_CHANCE = 40 # Probability of mutation in percentage
ITER_NUM = 5000 # Number of iterations (generations)
ELITE_PERCENT = 0.25 # Percentage of elite individuals to retain each generation
LOSS_TYPE = LOSS.DELTA # Type of loss function to use
IMAGE_NAME = 'FELV-cat' # Name of the target image

ELITE_NUMBER = int(np.ceil(ELITE_PERCENT * POPULATION_SIZE))
MULTIPLY_WEIGHTS = 50
COLOR_ME_TENDERS = True # if true color randomly chosen, else color is chosen by average color in target rectangle area
EDGE_THICKNESS = 0 # Thickness of the rectangle edges
MAX_SIZE = 20 # Maximum size of rectangles

# Directory to save generated images
directory = f"./images_genetic/{IMAGE_NAME}/{POPULATION_SIZE}_{RECTANGLE_AMOUNT}_{ITER_NUM}_" \
            f"{MAX_SIZE}_{EDGE_THICKNESS}_{MULTIPLY_WEIGHTS}_{ELITE_PERCENT}_{LOSS_TYPE}_{COLOR_ME_TENDERS}"


# Load a pre-trained VGG16 model
vgg = models.vgg16(weights="VGG16_Weights.DEFAULT").features


def calculate_perceptual_loss(curr_painting, target_img):
    """
    Calculate the perceptual loss between the current painting and the target image.
    The loss is computed based on features extracted from a pre-trained VGG16 model.
    @param curr_painting: The current generated image.
    @param target_img: The target image we want to approximate.
    @return: The perceptual loss between the two images.
    """
    img1 = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(curr_painting, cv2.COLOR_BGR2RGB)

    # Define transformations: resize, convert to tensor, normalize
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize images to 224x224 as VGG expects
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img1 = preprocess(img1).unsqueeze(0)  # Add batch dimension
    img2 = preprocess(img2).unsqueeze(0)  # Add batch dimension

    # Ensure the model is in evaluation mode
    vgg.eval()

    # Extract features
    def perceptual_extract_features(model, x):
        layers = [3, 8, 15, 22]  # Select layers at different depths
        features = []
        for i, layer in enumerate(model):
            x = layer(x)
            if i in layers:
                features.append(x)
        return features
    features_img1 = perceptual_extract_features(vgg, img1)
    features_img2 = perceptual_extract_features(vgg, img2)

    perceptual_loss = 0.0
    for f1, f2 in zip(features_img1, features_img2):
        perceptual_loss += torch.nn.functional.mse_loss(f1, f2)
    return perceptual_loss


def calculate_fitness_scores(population):
    """
    Calculate the fitness scores for a population of paintings.
    The fitness score is the negative of the loss, as we minimize loss.
    @param population: List of Painting objects representing the current population.
    @return: List of fitness scores.
    """
    fitness_scores = [-subject.loss for subject in population]
    return fitness_scores


def generate_distribution(fitness_scores):
    """
    Generate a probability distribution for selection based on fitness scores.
    @param fitness_scores: List of fitness scores.
    @return: Probability distribution for selection.
    """
    scores = np.abs(fitness_scores)
    probabilities = scores / np.sum(scores)
    return probabilities


def mutation_1(child, target_img):
    """
    Perform a mutation on the child's rectangles by replacing one of them with a new random rectangle.
    @param child: List of rectangles in the child painting.
    @param target_img: The original target image.
    @return: The mutated list of rectangles.
    """
    child = np.array(child, dtype=object)
    i = np.random.choice(child.shape[0], size=1, replace=False)
    child[i] = generate_random_rectangle(target_img=target_img, max_size=MAX_SIZE,
                                         edge_thickness=EDGE_THICKNESS, color_random=COLOR_ME_TENDERS)
    child = list(child)
    return child


def mutation_2(child, target_img):
    """
    Perform a mutation on the child's rectangles by adding a new random rectangle.
    @param child: List of rectangles in the child painting.
    @param target_img: The original target image.
    @return: The mutated list of rectangles with an additional rectangle.
    """
    child.append(generate_random_rectangle(target_img=target_img, max_size=MAX_SIZE,
                                           edge_thickness=EDGE_THICKNESS, color_random=COLOR_ME_TENDERS))
    return child


def reproduce(mom, dad, target_img):
    """
    Create a new child by combining rectangles from two parent paintings.
    The crossover is done either horizontally or vertically, followed by a possible mutation.
    @param mom: The mother Painting object.
    @param dad: The father Painting object.
    @param target_img: The target image to be approximated.
    @return: A new Painting object representing the child.
    """
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

    # Mutant me up scotty
    mutant_type = random.randint(1, 100)
    if mutant_type <= MUTATION_CHANCE/2:
        child_rectangles = mutation_1(child_rectangles, target_img)
    elif mutant_type <= MUTATION_CHANCE:
        child_rectangles = mutation_2(child_rectangles, target_img)
    return Painting(child_rectangles, target_img, mom.weights_matrix, LOSS_TYPE)


def genetic_algorithm(target_img, weights_matrix):
    """
    Run the genetic algorithm to evolve a population of paintings towards resembling the target image.
    @param target_img: The target image to be approximated.
    @return: The final sorted population and the list of losses per generation.
    """
    # Generate random population
    curr_population = np.empty(shape=(POPULATION_SIZE,), dtype=object)
    for i in range(POPULATION_SIZE):
        rectangle_list = random_rectangle_set(number_of_rectangles=RECTANGLE_AMOUNT, target_img=target_img,
                                              max_size=MAX_SIZE, edge_thickness=EDGE_THICKNESS,
                                              color_random=COLOR_ME_TENDERS)
        curr_population[i] = Painting(rectangle_list, target_img, weights_matrix, LOSS_TYPE)

    losses = []
    for i in tqdm(range(ITER_NUM)):
        curr_population = np.sort(curr_population) # sort by loss
        fitness_scores = calculate_fitness_scores(curr_population)
        distribution = generate_distribution(fitness_scores)

        if i % 5 == 0:
            # Save images every 5 generations for GIF
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
            curr_population[j] = reproduce(mom, dad, target_img)

    return np.sort(curr_population), losses


if __name__ == '__main__':
    original_image_path = f'layouts/{IMAGE_NAME}.jpg'
    original_image = cv2.imread(original_image_path)
    target_img = cv2.resize(original_image, PICTURE_SIZE, interpolation=cv2.INTER_AREA)

    weights_matrix = extract_features(target_img, MULTIPLY_WEIGHTS)

    last_generation, losses = genetic_algorithm(target_img, weights_matrix)
    losses.append(last_generation[0].loss)

    save_images_to_folder(directory+"/top5", [last_generation[i].image for i in range(5)], target_img,
                          bottom_texts=[f"Generation: {ITER_NUM}"] * 5)

    save_images_to_folder(directory + "/gif", [last_generation[0].image], target_img,
                          [f"Generation: {ITER_NUM}"], [f"image_{ITER_NUM}.png"])

    create_gif(directory + "/gif", directory + "/GIF.gif")

    plt.figure(figsize=(8, 7))  # Set the figure size
    plt.plot(losses, linestyle='-', color='blue')
    plt.title('Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(directory + "/loss.jpg")