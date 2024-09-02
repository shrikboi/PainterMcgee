import cv2
import numpy as np
import random
import imageio
import os
import matplotlib

matplotlib.use('TkAgg')
PRESENTING_SIZE = (512, 512)
PICTURE_SIZE = (128, 128)


class Rectangle(object):
    """
    A Rectangle represents a graphical element defined by its size, angle, edge thickness,
    center position, color, and opacity.
    """

    def __init__(self, size, angle, edge_thickness, center, color, opacity):
        self.size = size
        self.angle = angle
        self.edge_thickness = edge_thickness
        self.center = center
        self.color = color
        self.opacity = opacity

    def __eq__(self, other):
        return all([self.size == other.size, self.angle == other.angle, self.edge_thickness == other.edge_thickness,
                    self.center == other.center, self.color == other.color, self.opacity == other.opacity])


def draw_rectangle(image, rectangle):
    """
    Draw a rectangle on an image with the given parameters.
    @param image: The image on which to draw the rectangle.
    @param rectangle: The Rectangle object defining the rectangle's properties.
    @return: The image with the rectangle drawn.
    """
    rect = (rectangle.center, rectangle.size, rectangle.angle)

    box = cv2.boxPoints(rect)
    box = np.int32(box)
    overlay = image.copy()

    # Fill the polygon on the overlay with the specified color
    cv2.fillPoly(overlay, [box], rectangle.color)

    # Blend the overlay with the original image using the specified opacity
    cv2.addWeighted(overlay, rectangle.opacity, image, 1 - rectangle.opacity,
                    0, image)

    edge_thickness = rectangle.edge_thickness
    if edge_thickness != 0:
        cv2.drawContours(image, [box], 0, rectangle.color, edge_thickness)

    return image


def display_images_side_by_side(target_img, model_img, bottom_text=None):
    """
    Display two images side by side with optional text above each image.
    @param target_img: The first (left) image.
    @param model_img: The second (right) image.
    @param bottom_text: Optional text to display at the bottom.
    @return: A single image combining the input images side by side with text.
    """
    target_img = cv2.resize(target_img, PRESENTING_SIZE, interpolation=cv2.INTER_AREA)
    model_img = cv2.resize(model_img, PRESENTING_SIZE, interpolation=cv2.INTER_AREA)

    _, width1 = target_img.shape[:2]
    _, width2 = model_img.shape[:2]

    # Define the text and font
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    font_thickness = 1
    text_color = (255, 255, 255)

    # Add text above the images
    text1 = "Original Picture"
    text2 = "Model's Painting"

    # Create blank images for text
    text_img1 = np.zeros((50, width1, 3), dtype=np.uint8)
    text_img2 = np.zeros((50, width2, 3), dtype=np.uint8)

    # Put text on the blank images
    cv2.putText(text_img1, text1, (10, 30), font, font_scale, text_color,
                font_thickness, cv2.LINE_AA)
    cv2.putText(text_img2, text2, (10, 30), font, font_scale, text_color,
                font_thickness, cv2.LINE_AA)

    # Concatenate text images with the actual images
    image1_with_text = np.vstack((text_img1, target_img))
    image2_with_text = np.vstack((text_img2, model_img))

    # Concatenate the images side by side
    combined_image = np.hstack((image1_with_text, image2_with_text))
    if bottom_text is not None:
        bottom_padding = np.zeros((50, combined_image.shape[:2][1], 3), dtype=np.uint8)
        cv2.putText(bottom_padding, bottom_text, (10, 30), font, font_scale, text_color,
                    font_thickness, cv2.LINE_AA)

        return np.vstack((combined_image, bottom_padding))

    # Display the combined image
    return combined_image


def extract_features(image, multiply_weights):
    """
    Extract SIFT features from an image and create a heatmap representing feature density.
    @param image: The input image.
    @param multiply_weights: A multiplier applied to the heatmap for weighting features.
    @return: A heatmap representing the distribution of SIFT features in the image.
    """
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints
    keypoints = sift.detect(gray_image, None)

    # Create an empty image with the same dimensions as original image
    heatmap = np.zeros_like(gray_image, dtype=np.float32)

    # Accumulate points in the heatmap
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        heatmap[y, x] += 1

    # Blur the heatmap to spread out the "heat"
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=4, sigmaY=4)

    # Normalize the heatmap
    if heatmap.min() < 0:
        heatmap += heatmap.min()
    heatmap = heatmap * multiply_weights
    if heatmap.min() < 1:
        heatmap += 1 - heatmap.min()

    return heatmap


def extract_roi(rect, target_image):
    """
    Extract the Region of Interest (ROI) defined by a rotated rectangle from the target image.
    @param rect: A tuple representing the rectangle (center, size, angle).
    @param target_image: The image from which to extract the ROI.
    @return: The extracted ROI as an image.
    """
    # Calculate the bounding box of the rotated rectangle in original coordinates
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # Get the coordinates of the bounding box
    x, y, w, h = cv2.boundingRect(box)

    # Ensure the bounding box is within the image boundaries
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, target_image.shape[1] - x)
    h = min(h, target_image.shape[0] - y)

    # Extract the bounding box region from the original image
    return target_image[y:y + h, x:x + w]


def calculate_color(rectangle_size, rectangle_center, rectangle_angle, target_img):
    """
    Calculate the average color of the region defined by a rectangle in the original image.
    @param rectangle_size: The size (width, height) of the rectangle.
    @param rectangle_center: The center (x, y) of the rectangle.
    @param rectangle_angle: The rotation angle of the rectangle.
    @param target_img: The original image.
    @return: The average color of the region as a BGR tuple.
    """
    rect = (rectangle_center, rectangle_size, rectangle_angle)

    bounding_box_roi = extract_roi(rect, target_img)

    # Calculate the center of the bounding box region
    bounding_box_center = (
        bounding_box_roi.shape[1] // 2, bounding_box_roi.shape[0] // 2)

    # Get the rotation matrix for the bounding box region
    M = cv2.getRotationMatrix2D(bounding_box_center, rectangle_angle, 1.0)

    # Rotate the bounding box region
    rotated_roi = cv2.warpAffine(bounding_box_roi, M, (
        bounding_box_roi.shape[1], bounding_box_roi.shape[0]))

    # Calculate the new center of the rotated ROI
    new_center = (rotated_roi.shape[1] // 2, rotated_roi.shape[0] // 2)
    rect = (new_center, rectangle_size,
            0)  # Angle is 0 because we already rotated the region

    # Calculate the bounding box of the rotated rectangle in the rotated ROI coordinates
    final_roi = extract_roi(rect, rotated_roi)

    # Calculate the average color
    avg_color = cv2.mean(final_roi)[:3]  # Exclude the alpha channel if present
    return avg_color


def random_rectangle_set(number_of_rectangles, target_img, max_size, edge_thickness, color_random):
    """
    Generate a set of random rectangles.
    @param number_of_rectangles: The number of rectangles to generate.
    @param target_img: The target image for color calculation if color_random is False.
    @param max_size: The maximum size of the rectangles.
    @param edge_thickness: The thickness of the rectangle's edges.
    @param color_random: If True, colors are chosen randomly; otherwise, they are based on the target image.
    @return: A list of randomly generated Rectangle objects.
    """
    return [generate_random_rectangle(target_img=target_img, max_size=max_size,
                                      edge_thickness=edge_thickness,
                                      color_random=color_random) for _ in range(number_of_rectangles)]


def generate_random_rectangle(target_img, max_size, edge_thickness=0, color_random=False):
    """
    Generate a random rectangle with specified parameters.
    @param target_img: The target image for color calculation if color_random is False.
    @param max_size: The maximum size of the rectangle.
    @param edge_thickness: The thickness of the rectangle's edges.
    @param color_random: If True, colors are chosen randomly; otherwise, they are based on the target image.
    @return: A Rectangle object with random properties.
    """
    width = random.randint(0, max_size)
    height = random.randint(0, max_size)
    rectangle_size = (width, height)
    rectangle_angle = random.randint(0, 359)  # Random degree between 0 and 360
    center_width = random.randint(0, PICTURE_SIZE[0] - 1)
    center_height = random.randint(0, PICTURE_SIZE[1] - 1)
    rectangle_center = (center_width, center_height)
    if color_random:
        rectangle_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    else:
        rectangle_color = calculate_color(rectangle_size, rectangle_center, rectangle_angle, target_img)

    rectangle_opacity = (random.randint(1, 10)) / 10

    if edge_thickness:  # Random edge_thickness between 0 and 1
        edge_thickness = random.randint(0, edge_thickness)

    rectangle = Rectangle(rectangle_size, rectangle_angle, edge_thickness,
                          rectangle_center, rectangle_color, rectangle_opacity)

    return rectangle


def save_images_to_folder(directory, images, target_img, bottom_texts=None, titles=None):
    """
    Save a list of images to a folder with optional titles and bottom texts.
    @param directory: The directory where the images will be saved.
    @param images: List of images to be saved.
    @param target_img: The original image to display alongside the generated images.
    @param bottom_texts: Optional list of texts to display below each image.
    @param titles: Optional list of titles for each image.
    @return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, image in enumerate(images):
        title = titles[i] if titles is not None else f"image_{i}"
        bottom_text = bottom_texts[i] if bottom_texts is not None else None
        cv2.imwrite(os.path.join(directory, title + ".png"),
                    display_images_side_by_side(target_img, image, bottom_text))


def create_gif(image_folder, output_path, duration=0.2):
    """
    Create a GIF from a series of images in a folder.
    @param image_folder: The folder containing the images.
    @param output_path: The path where the GIF will be saved.
    @param duration: The duration of each frame in the GIF.
    @return:
    """
    def sort_key(filename):
        parts = filename.split('_')  # Split by underscore
        number_part = parts[-1].split('.')[0]  # Get the number part and remove file extension
        return int(number_part)  # Convert to integer for correct numerical sorting

    images = []
    file_list = os.listdir(image_folder)
    file_list_sorted = sorted(file_list, key=sort_key)
    for filename in file_list_sorted:
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            # Convert BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    # Write images to a gif
    imageio.mimsave(output_path, images, duration=duration)


def add_gaussian_noise(rgb_tuple, mean=0, stddev=5):
    """
    Add Gaussian noise to an RGB color tuple.
    @param rgb_tuple: A tuple representing an RGB color.
    @param mean: The mean of the Gaussian noise.
    @param stddev: The standard deviation of the Gaussian noise.
    @return: A tuple representing the noisy RGB color.
    """
    noisy_rgb = []
    for value in rgb_tuple:
        noise = np.random.normal(mean, stddev)
        noisy_value = value + noise
        # Clamp the value between 0 and 1
        noisy_value = min(max(noisy_value, 0), 255)
        noisy_rgb.append(noisy_value)

    return tuple(noisy_rgb)


def log_losses(directory, losses):
    """
    Log the loss values to a text file.
    @param directory: The directory where the log file will be saved.
    @param losses: A list of loss values.
    @return:
    """
    log_file_path = os.path.join(directory, "loss_log.txt")

    with open(log_file_path, "w") as log_file:
        for iteration, loss in enumerate(losses):
            log_file.write(f"Iteration: {iteration}, Loss: {loss}\n")

    print(f"Loss log saved to {log_file_path}")

