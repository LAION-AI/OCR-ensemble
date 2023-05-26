import numpy as np
import cv2
from PIL import Image
import math

def applyPadding(xywhbbox, padding, xmax, ymax):
    x, y, w, h = xywhbbox 
    x -= padding 
    y -= padding 
    w += 2*padding 
    h += 2*padding
    # clip to the image size
    x = max(0, x)
    y = max(0, y)
    if x+w > xmax: 
        w -= (x + w) - xmax 
    if y+h > ymax:
        h -= (y+h) - ymax
    return [x, y, w, h]

                
def xywh2uull(xywhbbox):
    # [22.0, 168.0], [235.0, 168.0], [235.0, 196.0], [22.0, 196.0] is an example paddle bounding box 
    x, y, w, h = xywhbbox
    upper_left = [x, y]
    upper_right = [x+w, y]
    lower_left = [x, y+h]
    lower_right = [x+w, y+h]
    return [upper_left, upper_right, lower_right, lower_left]

def uull2xywh(uullbbox):
    # copy-paste from: https://stackoverflow.com/questions/74151879/how-to-crop-the-specific-part-of-image-using-paddleocr-bounding-box-co-ordinates
    box = np.array(uullbbox).astype(np.int32).reshape(-1, 2)
    points = np.array([box])
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    return rect


def rotatedCrop(img, corners, flip_if_vertical=False, return_180=False, return_90s=False):
    image = Image.fromarray(img)
    #code written by chatgpt4
    upper_left, upper_right, lower_right, lower_left = corners

    # Calculate the width and height of the unrotated box
    width = int(np.linalg.norm(np.array(upper_right) - np.array(upper_left)))
    height = int(np.linalg.norm(np.array(upper_left) - np.array(lower_left)))

    # Calculate the center of the box
    center_x = int((upper_left[0] + lower_right[0]) / 2)
    center_y = int((upper_left[1] + lower_right[1]) / 2)

    # Calculate the rotation angle of the box
    dx = upper_right[0] - upper_left[0]
    dy = upper_right[1] - upper_left[1]
    rotation_angle = math.degrees(math.atan2(dy, dx))
    # Rotate the image back by rotation_angle
    rotated_image = image.rotate(rotation_angle, resample=Image.BICUBIC, center=(center_x, center_y))

    # Crop the unrotated box
    left = center_x - width // 2
    upper = center_y - height // 2
    right = center_x + width // 2
    lower = center_y + height // 2

    if flip_if_vertical and height > 1.1*width:
        # Rotate again to make the text horizontal
        rotated_image = rotated_image.rotate(90, resample=Image.BICUBIC, center=(center_x, center_y))
        # Swap width and height
        width, height = height, width 
        # Crop the unrotated box
        left = center_x - width // 2
        upper = center_y - height // 2
        right = center_x + width // 2
        lower = center_y + height // 2

    cropped_image = rotated_image.crop((left, upper, right, lower))
    if return_180:
        cropped_center_x, cropped_center_y = cropped_image.width // 2, cropped_image.height // 2
        return cropped_image, cropped_image.rotate(180, resample=Image.BICUBIC, center=(cropped_center_x, cropped_center_y))
    if return_90s:
        cropped_center_x, cropped_center_y = cropped_image.width // 2, cropped_image.height // 2
        cropped_image = np.array(cropped_image)
        r90 = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
        r180 = cv2.rotate(r90, cv2.ROTATE_90_CLOCKWISE)
        r270 = cv2.rotate(r180, cv2.ROTATE_90_CLOCKWISE)
        return cropped_image, r90, r180, r270
    return np.array(cropped_image)

