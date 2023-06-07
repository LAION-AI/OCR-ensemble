from weighted_levenshtein import lev
import numpy as np
import cv2

def levenshtein(gt, pred):
    return lev(gt.encode("ascii", "ignore").decode(), pred.encode("ascii", "ignore").decode())

def levenshtein_lower(gt, pred):
    return lev(gt.lower().encode("ascii", "ignore").decode(), pred.lower().encode("ascii", "ignore").decode())

def average_levenshtein(gts, preds, dist=levenshtein):
    levs = []
    for gt, pred in zip(gts, preds):
        levs += [dist(gt, pred)]
    return np.mean(levs)

def create_mask(bounding_boxes, mask_size):
    mask = np.zeros(mask_size)
    for box in bounding_boxes:
        # Convert the box points to integer type
        box_int = np.array(box, dtype=np.int32)
        # Reshape to match the shape expected by cv2.fillPoly (must be 3-dimensional)
        reshaped = box_int.reshape((-1, 1, 2))
        # Draw the rotated rectangle on the mask
        cv2.fillPoly(mask, [reshaped], color=255)
    return mask

def intersection_mask(mask1, mask2):
    return np.logical_and(mask1, mask2)

def union_mask(mask1, mask2):
    return np.logical_or(mask1, mask2)

def iou_score(bb_gt, bb_predict, image_width, image_height):
    bb1 = bb_gt
    bb2 = bb_predict
    w = image_width
    h = image_height
    bb1_mask = create_mask(bb1, (h, w))
    bb2_mask = create_mask(bb2, (h, w))
    intersection = intersection_mask(bb1_mask, bb2_mask)
    union = union_mask(bb1_mask, bb2_mask)
    return np.sum(intersection) / np.sum(union)