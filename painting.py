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
            for i in range(0, self.size[0], 1):
                for j in range(0, self.size[1], 1):
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
        cpy_painting.original_image = np.copy(self.original_image)
        cpy_painting.rectangle_list = np.copy(self.rectangle_list)
        cpy_painting.size = np.copy(self.size)
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

    def calculate_average_color(self):
        rect = (self.center, self.rectangle.size, self.rectangle.angle)

        # Calculate the bounding box of the rotated rectangle in original coordinates
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Get the coordinates of the bounding box
        x, y, w, h = cv2.boundingRect(box)

        # Ensure the bounding box is within the image boundaries
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, self.original_picture.shape[1] - x)
        h = min(h, self.original_picture.shape[0] - y)

        # Extract the bounding box region from the original image
        bounding_box_roi = self.original_picture[y:y + h, x:x + w]

        # Calculate the center of the bounding box region
        bounding_box_center = (bounding_box_roi.shape[1] // 2, bounding_box_roi.shape[0] // 2)

        # Get the rotation matrix for the bounding box region
        M = cv2.getRotationMatrix2D(bounding_box_center, self.rectangle.angle, 1.0)

        # Rotate the bounding box region
        rotated_roi = cv2.warpAffine(bounding_box_roi, M,
                                     (bounding_box_roi.shape[1], bounding_box_roi.shape[0]))

        # Calculate the new center of the rotated ROI
        new_center = (rotated_roi.shape[1] // 2, rotated_roi.shape[0] // 2)

        # Calculate the bounding box of the rotated rectangle in the rotated ROI coordinates
        rect = (new_center, self.rectangle.size, 0)  # Angle is 0 because we already rotated the region
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        x, y, w, h = cv2.boundingRect(box)

        # Ensure the ROI is within the rotated region boundaries
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, rotated_roi.shape[1] - x)
        h = min(h, rotated_roi.shape[0] - y)

        # Extract the final ROI from the rotated bounding box region
        final_roi = rotated_roi[y:y + h, x:x + w]

        # Calculate the average color
        avg_color = cv2.mean(final_roi)[:3]  # Exclude the alpha channel if present
        return avg_color

