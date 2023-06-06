from weighted_levenshtein import lev
import numpy as np

def levenshtein(gt, pred):
    return lev(gt.encode("ascii", "ignore").decode(), pred.encode("ascii", "ignore").decode())

def levenshtein_lower(gt, pred):
    return lev(gt.lower().encode("ascii", "ignore").decode(), pred.lower().encode("ascii", "ignore").decode())

def average_levenshtein(gts, preds, dist=levenshtein):
    levs = []
    for gt, pred in zip(gts, preds):
        levs += [dist(gt, pred)]
    return np.mean(levs)

