import numpy as np
from utils import draw_rectangle
import cv2
import colour
from skimage.metrics import structural_similarity as ssim

PICTURE_SIZE = (128, 128)


class Painting(object):
    def __init__(self, rectangle_list, target_image, weights_matrix, loss_type, ssim_weight=0.5):
        self.rectangle_list = rectangle_list
        self.weights_matrix = weights_matrix
        self.target_image = target_image
        self.image = self.draw_rectangles()
        if loss_type == "delta":
            self.loss = self.delta_e_loss()
        elif loss_type == "mse":
            self.loss = self.mse_loss()
        elif loss_type == "ssim":
            self.loss = self.ssim_loss()
        elif loss_type == "deltassim":
            self.loss = self.delta_e_loss()*(1-ssim_weight) + self.ssim_loss()*ssim_weight
        elif loss_type == "mserssim":
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
        return np.mean(delta_e * self.weights_matrix)

    def mse_loss(self):
        weights_stacked = np.stack([self.weights_matrix, self.weights_matrix, self.weights_matrix], axis=-1)
        return np.mean(np.square(self.image - self.target_image) * weights_stacked)

    def ssim_loss(self):
        return -ssim(self.image, self.target_image, channel_axis=2, data_range=225)

    def __lt__(self, other):
        return self.loss < other.loss
