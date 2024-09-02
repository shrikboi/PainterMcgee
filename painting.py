import numpy as np
from utils import draw_rectangle
import cv2
import colour
from skimage.metrics import structural_similarity as ssim
from enum import Enum
PICTURE_SIZE = (128, 128)


class LOSS(Enum):
    DELTA = 1
    MSE = 2
    SSIM = 3
    DELTASSIM = 4
    MSESSIM = 5

class Painting(object):
    def __init__(self, rectangle_list, target_img, weights_matrix, loss_type,
                 ssim_weight=0.5):
        self.rectangle_list = rectangle_list
        self.weights_matrix = weights_matrix
        self.target_image = target_img
        self.image = self.draw_rectangles()
        self.feature_extract = True if loss_type in [LOSS.DELTA, LOSS.MSE] else False
        if loss_type == LOSS.DELTA:
            self.loss = self.delta_e_loss()
        elif loss_type == LOSS.MSE:
            self.loss = self.mse_loss()
        elif loss_type == LOSS.SSIM:
            self.loss = self.ssim_loss()
        elif loss_type == LOSS.DELTASSIM:
            self.loss = self.delta_e_loss()*(1-ssim_weight) + self.ssim_loss()*ssim_weight
        elif loss_type == LOSS.MSESSIM:
            self.loss = self.mse_loss()*(1-ssim_weight) + self.ssim_loss()*ssim_weight

    def draw_rectangles(self):
        painting = np.ones((*PICTURE_SIZE[::-1], 3), dtype=np.uint8) * 255
        for rectangle in self.rectangle_list:
            draw_rectangle(painting, rectangle)
        return painting

    def delta_e_loss(self):
        target_image = cv2.cvtColor(self.target_image.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
        image = cv2.cvtColor(self.image.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)

        delta_e = colour.difference.delta_e.delta_E_CIE2000(target_image, image)
        if self.feature_extract:
            final = np.mean(delta_e * self.weights_matrix)
        else:
            final = np.mean(delta_e)
        return final

    def mse_loss(self):
        weights_stacked = np.stack([self.weights_matrix, self.weights_matrix, self.weights_matrix],
                                   axis=-1)
        squared_diff = np.square(self.image.astype(np.float32) - self.target_image.astype(np.float32))
        euclidean_distance = np.sqrt(np.sum(squared_diff, axis=-1))
        if self.feature_extract:
            final = np.mean(euclidean_distance * self.weights_matrix)
        else:
            final = np.mean(euclidean_distance)
        return final

    def ssim_loss(self):
        return -ssim(self.image, self.target_image, channel_axis=2, data_range=225)

    def __lt__(self, other):
        return self.loss < other.loss


class BeamSearch_Node(Painting):
    def __init__(self, rectangle_list, target_img, weights_matrix, loss_type=LOSS_TYPE, parent=None):
        super().__init__(rectangle_list, target_img, weights_matrix, loss_type)
        self.parent = parent

    def get_path(self):
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]  # return reversed path