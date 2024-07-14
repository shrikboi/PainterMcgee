import cv2
import numpy as np
import random

PRESENTING_SIZE = (512, 512)


def draw_rectangle(image, center, rectangle, color=(300, 41, 62)):
    """
    Draw a rectangle on an image with the given parameters.

    :param image: The image on which to draw the rectangle.
    :param center: The center of the rectangle (x, y).
    :param rectangle: The rectangle.
    :param color: The color of the rectangle (default is black).
    :return: Image with the rectangle drawn.
    """
    rect = (center, rectangle.size, rectangle.angle)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.fillPoly(image, [box], color)
    edge_thickness = rectangle.edge_thickness
    if edge_thickness != 0:
        cv2.drawContours(image, [box], 0, (0, 0, 0), rectangle.edge_thickness)

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
