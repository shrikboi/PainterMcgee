import cv2
import numpy as np
import random

PRESENTING_SIZE = (512, 512)
PICTURE_SIZE = (128, 128)
MAX_REC_WIDTH = 30
MAX_REC_HEIGHT = 30


class Rectangle(object):
    """
    A Rectangle is a collection of

    """

    def __init__(self, size, angle, edge_thickness, center, color, opacity):
        self.size = size
        self.angle = angle
        self.edge_thickness = 0
        self.center = center
        self.color = color
        self.opacity = 0.3


def draw_rectangle(image, rectangle, color=(300, 41, 62)):
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
    cv2.addWeighted(overlay, rectangle.opacity, image, 1 - rectangle.opacity, 0, image)

    edge_thickness = rectangle.edge_thickness
    if edge_thickness != 0:
        cv2.drawContours(image, [box], 0, rectangle.color, rectangle.edge_thickness)

    return image


def display_images_side_by_side(image1, image2):
    """
    Display two images side by side with text above each image.

    :param image1: First (left) image.
    :param image2: Second (right) image.
    """
    image1 = cv2.resize(image1, PRESENTING_SIZE, interpolation=cv2.INTER_AREA)
    image2 = cv2.resize(image2, PRESENTING_SIZE, interpolation=cv2.INTER_AREA)

    _, width1 = image1.shape[:2]
    _, width2 = image2.shape[:2]

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
    cv2.putText(text_img1, text1, (10, 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(text_img2, text2, (10, 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Concatenate text images with the actual images
    image1_with_text = np.vstack((text_img1, image1))
    image2_with_text = np.vstack((text_img2, image2))

    # Concatenate the images side by side
    combined_image = np.hstack((image1_with_text, image2_with_text))

    # Display the combined image
    return combined_image


def draw_all_rectangles(rectangle_list, size):
    """
    Draw all possible rectangles one by one to help us see if the rectangles are good :)
    """
    for i, rectangle in enumerate(rectangle_list):
        image = np.ones((*size, 3), dtype=np.uint8) * 255
        image = draw_rectangle(image, (size[0]/2, size[1]/2), rectangle)
        cv2.imshow(f'rectangle {i}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def generate_rectangles(n, max_width, max_height, with_edges):
    with open("./layouts/rectangle_list.txt", "w") as file:
        for _ in range(n):
            width = random.randint(1, max_width)  # Random width between 1 and 100
            height = random.randint(1, max_height)  # Random height between 1 and 100
            degree = random.randint(0, 360)  # Random degree between 0 and 360
            if with_edges:
                edge_thickness = random.randint(1, 4)  # Random edge_thickness between 1 and 10
            else:
                edge_thickness = 0  # Edge thickness is zero if edge is False

            file.write(f"{width} {height} {degree} {edge_thickness}\n")


def extract_features(image):
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
    heatmap = heatmap * 10
    if heatmap.min() < 1:
        heatmap += 1-heatmap.min()
    return heatmap

    # Convert heatmap to color (for visualization)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Display the heatmap
    cv2.imshow('Heatmap', heatmap_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_roi(rect, og_image):
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
    return og_image[y:y + h, x:x + w]


def calculate_color(rectangle_size, rectangle_center, rectangle_angle, og_image):
    rect = (rectangle_center, rectangle_size, rectangle_angle)

    bounding_box_roi = extract_roi(rect, og_image)

    # Calculate the center of the bounding box region
    bounding_box_center = (bounding_box_roi.shape[1] // 2, bounding_box_roi.shape[0] // 2)

    # Get the rotation matrix for the bounding box region
    M = cv2.getRotationMatrix2D(bounding_box_center, rectangle_angle, 1.0)

    # Rotate the bounding box region
    rotated_roi = cv2.warpAffine(bounding_box_roi, M, (bounding_box_roi.shape[1], bounding_box_roi.shape[0]))

    # Calculate the new center of the rotated ROI
    new_center = (rotated_roi.shape[1] // 2, rotated_roi.shape[0] // 2)
    rect = (new_center, rectangle_size, 0)  # Angle is 0 because we already rotated the region

    # Calculate the bounding box of the rotated rectangle in the rotated ROI coordinates
    final_roi = extract_roi(rect, rotated_roi)

    # Calculate the average color
    avg_color = cv2.mean(final_roi)[:3]  # Exclude the alpha channel if present
    return avg_color


def generate_random_rectangle(og_image, edge_thickness=0):
    width = random.randint(1, MAX_REC_WIDTH)  # Random width between 1 and 100
    height = random.randint(1, MAX_REC_HEIGHT)  # Random height between 1 and 100
    rectangle_size = (width, height)
    rectangle_angle = random.randint(0, 360)  # Random degree between 0 and 360
    center_width = random.randint(0, PICTURE_SIZE[0])
    center_height = random.randint(0, PICTURE_SIZE[1])
    rectangle_center = (center_width, center_height)
    rectangle_color = calculate_color(rectangle_size, rectangle_center, rectangle_angle, og_image)
    rectangle_opacity = (random.randint(1, 4))/10

    if edge_thickness:
        edge_thickness = random.randint(1, edge_thickness)  # Random edge_thickness between 1 and 10

    rectangle = Rectangle(rectangle_size, rectangle_angle, edge_thickness, rectangle_center, rectangle_color,
                          rectangle_opacity)

    return rectangle