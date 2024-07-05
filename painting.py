import numpy as np
from utils import draw_rectangle
import cv2
PICTURE_SIZE = (128,128) #(height, width)


class Painting:

    """

    """

    def __init__(self, original_image, rectangle_list, current_painting=None):
        self.original_image = original_image
        self.rectangle_list = rectangle_list
        if current_painting is None:
            self.current_painting = np.ones((*(PICTURE_SIZE), 3), dtype=np.uint8) * 255
        else:
            self.current_painting = current_painting

    def add_move(self, move):
        """
        Try to add <player>'s <move>.

        If the move is legal, the board state is updated; if it's not legal, a
        ValueError is raised.

        Returns the number of tiles placed on the board.
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
            for i in range(0, PICTURE_SIZE[0], 10):
                for j in range(0, PICTURE_SIZE[1], 10):
                    legal_moves.append(Move(rectangle,(i,j),self.original_image))
        return legal_moves

    def score(self):
        squared_diff = (self.original_image - self.current_painting) ** 2
        # Calculate the mean of the squared differences
        mean_squared_diff = np.mean(squared_diff)
        return mean_squared_diff

    def __eq__(self, other):
        return np.array_equal(self.current_painting, other.current_painting)


    def __copy__(self):
        cpy_painting = Painting(self.original_image, self.rectangle_list, self.current_painting)
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
        self.color = self.decide_color() #TODO fucntion to decide color

    def decide_color(self):
        return self.calculate_average_color()

    def extract_roi(self):
        """
        Extract the region of interest (ROI) defined by a rotated rectangle from the image.

        :param image: The input image.
        :param center: The center of the rectangle (x, y).
        :param size: The size of the rectangle (width, height).
        :param angle: The rotation angle of the rectangle in degrees.
        :return: The extracted ROI.
        """

        rect = (self.center[::-1], self.rectangle.size[::-1], self.rectangle.angle)

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(self.center, self.rectangle.angle, 1.0)

        # Rotate the entire image
        rotated_image = cv2.warpAffine(self.original_picture, M,
                                       (self.original_picture.shape[1], self.original_picture.shape[0]))

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

        :param image: The input image.
        :param center: The center of the rectangle (x, y).
        :param size: The size of the rectangle (width, height).
        :param angle: The rotation angle of the rectangle in degrees.
        :return: The average color (BGR).
        """
        # Extract the ROI
        roi = self.extract_roi()
        # Calculate the average color
        avg_color = cv2.mean(roi)[:3]  # Exclude the alpha channel if present

        return avg_color

