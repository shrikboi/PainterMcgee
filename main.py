from utils import draw_all_rectangles, display_images_side_by_side, generate_rectangles
from rectangle import RectangleList
from search import not_a_real_search
from paint import Paint
import cv2

PICTURE_SIZE = (128, 128) #(width, height)


def main():
    #load original picture
    original_image_path = 'layouts/image.jpg'
    original_image = cv2.imread(original_image_path)
    resized_image = cv2.resize(original_image, PICTURE_SIZE, interpolation=cv2.INTER_AREA)


    #load any legal moves
    rectangle_list = RectangleList("rectangle_list.txt")

    paint = Paint(resized_image, rectangle_list, PICTURE_SIZE)
    final_painting = not_a_real_search(paint)
    display_images_side_by_side(resized_image, final_painting)


def check_our_rectangle_list():
    rectangle_list = RectangleList("rectangle_list.txt")
    draw_all_rectangles(rectangle_list, PICTURE_SIZE)


generate_rectangles(60, 20, 20, False)
main()
# check_our_rectangle_list()
