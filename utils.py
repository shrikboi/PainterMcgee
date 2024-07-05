import cv2
import numpy as np


def draw_rectangle(image, center, rectangle, color=(0, 0, 0)):
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
    cv2.drawContours(image, [box], 0, (0, 0, 0), rectangle.edge_thickness)

    return image


def display_images_side_by_side(image1, image2):
    """
    Display two images side by side with text above each image.

    :param image1: First (left) image.
    :param image2: Second (right) image.
    """

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
    cv2.imshow('Images Side by Side', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
