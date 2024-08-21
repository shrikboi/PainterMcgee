import cv2
import numpy as np
import random
import imageio
import os

PRESENTING_SIZE = (512, 512)
# PRESENTING_SIZE = (300, 400)
PICTURE_SIZE = (128, 128)


class Rectangle(object):
    """
    A Rectangle is a collection of

    """

    def __init__(self, size, angle, edge_thickness, center, color, opacity):
        self.size = size
        self.angle = angle
        self.edge_thickness = edge_thickness
        self.center = center
        self.color = color
        self.opacity = opacity

    def __lt__(self, other):
        return self.size[0]*self.size[1] < other.size[0]*other.size[1]


def draw_rectangle(image, rectangle):
    """
    Draw a rectangle on an image with the given parameters.

    :param image: The image on which to draw the rectangle.
    :param rectangle: The rectangle.
    :param color: The color of the rectangle (default is black).
    :return: Image with the rectangle drawn.
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


def display_images_side_by_side(target_img, model_img, bottom_text):
    """
    Display two images side by side with text above each image.

    :param target_img: First (left) image.
    :param model_img: Second (right) image.
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


def display_image(image):
    return cv2.resize(image, PRESENTING_SIZE, interpolation=cv2.INTER_AREA)


def extract_features(image, multiply_weights):
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

    # Normalize the heatmap to sum to 1
    # heatmap /= np.sum(heatmap)
    if heatmap.min() < 0:
        heatmap += heatmap.min()
    heatmap = heatmap * multiply_weights
    if heatmap.min() < 1:
        heatmap += 1 - heatmap.min()
    return heatmap


def extract_roi(rect, target_image):
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


def calculate_color(rectangle_size, rectangle_center, rectangle_angle, og_image):
    rect = (rectangle_center, rectangle_size, rectangle_angle)

    bounding_box_roi = extract_roi(rect, og_image)

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


def generate_random_init(number_of_rectangles, target_image, max_size, edge_thickness, color_random):
    return [generate_random_rectangle(target_image=target_image, max_size=max_size,
                                      edge_thickness=edge_thickness,
                                      color_random=color_random) for _ in range(number_of_rectangles)]


def generate_random_rectangle(target_image, max_size, edge_thickness=0, color_random=False):
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
        rectangle_color = calculate_color(rectangle_size, rectangle_center, rectangle_angle, target_image)
        rectangle_color = add_gaussian_noise(rectangle_color)

    rectangle_opacity = (random.randint(1, 10)) / 10

    if edge_thickness: # Random edge_thickness between 0 and 1
        edge_thickness = random.randint(0, edge_thickness)

    rectangle = Rectangle(rectangle_size, rectangle_angle, edge_thickness,
                          rectangle_center, rectangle_color, rectangle_opacity)

    return rectangle


def save_paintings_to_folder(directory, images, target_image, bottom_text):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, image in enumerate(images):
        cv2.imwrite(
            os.path.join(directory, f"image_{i}.png"),
            display_images_side_by_side(target_image, image, bottom_text))


def save_image_to_folder(directory, image, name):
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(os.path.join(directory, name), image)


def create_gif(image_folder, output_path, duration=2):
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



# original_image_path = 'layouts/chair.jpg'
# question_path = 'layouts/question.jpg'
# original_image = cv2.imread(original_image_path)
# model_image = cv2.imread(question_path)
# final = display_images_side_by_side(target_img=original_image, model_img=model_image, iteration_num=None)
# cv2.imwrite('./question.png', final)