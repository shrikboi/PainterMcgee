import numpy as np
from utils import draw_rectangle
import cv2


class Painting:
    """

    """

    def __init__(self, original_image, rectangle_list, size, current_painting=None):
        self.original_image = original_image
        self.rectangle_list = rectangle_list
        self.size = size
        if current_painting is None:
            self.current_painting = np.ones((*size[::-1], 3), dtype=np.uint8) * 255
        else:
            self.current_painting = current_painting.copy()

    def add_move(self, move):
        """
        Try to add move
        """
        self.current_painting = draw_rectangle(self.current_painting, move.center, move.rectangle, move.color)
        return 1

    def do_move(self, move):
        """
        Performs a move, returning a new board
        """
        new_painting = self.__copy__()
        new_painting.add_move(move)
        return new_painting

    def get_legal_moves(self):
        """
        Returns a list of legal moves for given player for this board state
        """
        # Generate all legal moves
        legal_moves = []
        for rectangle in self.rectangle_list:
            for i in range(0, self.size[0], 10):
                for j in range(0, self.size[1], 10):
                    legal_moves.append(Move(rectangle, (i, j), self.original_image))
        return legal_moves

    def score(self):
        squared_diff = (self.original_image - self.current_painting) ** 2
        # Calculate the mean of the squared differences
        mean_squared_diff = np.mean(squared_diff)
        return mean_squared_diff

    def __eq__(self, other):
        return np.array_equal(self.current_painting, other.current_painting)

    def __copy__(self):
        cpy_painting = Painting(self.original_image, self.rectangle_list, self.size, self.current_painting)
        cpy_painting.rectangle_list = np.copy(self.rectangle_list)
        cpy_painting.original_image = np.copy(self.original_image)
        cpy_painting.current_painting = np.copy(self.current_painting)
        return cpy_painting


class Move:
    """
    A Move describes how one of the players is going to spend their move.

    It contains:

    """

    def __init__(self, rectangle, center, original_picture):
        self.rectangle = rectangle
        self.center = center
        self.original_picture = original_picture
        self.color = self.calculate_average_color()

    def extract_roi(self):
        """
        Extract the region of interest (ROI) defined by a rotated rectangle from the image.
        :return: The extracted ROI.
        """

        rect = (self.center, self.rectangle.size, self.rectangle.angle)
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(self.center, self.rectangle.angle, 1.0)

        # Rotate the entire image
        rotated_image = cv2.warpAffine(self.original_picture, M, (self.original_picture.shape[:2])[::1])

        # Calculate the bounding box of the rotated rectangle
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Get the coordinates of the bounding box
        x, y, w, h = cv2.boundingRect(box)

        # Extract the ROI from the rotated image
        roi = rotated_image[y:y + h, x:x + w]

        return roi

    def calculate_average_color(self):
        """
        Calculate the average color of the region defined by a rotated rectangle in the image.
        :return: The average color (BGR).
        """
        # Extract the ROI
        roi = self.extract_roi()
        # Calculate the average color
        avg_color = cv2.mean(roi)[:3]  # Exclude the alpha channel if present

        return avg_color
