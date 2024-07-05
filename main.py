from utils import draw_rectangle, resize_image, display_images_side_by_side
from rectangle import RectangleList
from search import not_a_real_search
from paint import Paint
import cv2

PICTURE_SIZE = (128,128) #(height, width)

#load original picture
original_image_path = 'layouts/image.jpg'
original_image = cv2.imread(original_image_path)
resized_image = cv2.resize(original_image, PICTURE_SIZE[::-1], interpolation=cv2.INTER_AREA)

#load any legal moves
rectangle_list = RectangleList("rectangle_list.txt")

paint = Paint(resized_image,rectangle_list)
final_painting = not_a_real_search(paint)
display_images_side_by_side(resized_image, final_painting)

