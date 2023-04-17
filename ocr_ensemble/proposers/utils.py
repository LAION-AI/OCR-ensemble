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


def rotatedCrop(img, corners):
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
    cropped_image = rotated_image.crop((left, upper, right, lower))
    return np.array(cropped_image)

