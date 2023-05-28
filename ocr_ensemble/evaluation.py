from weighted_levenshtein import lev
import numpy as np

def average_levenshtein(gts, preds):
    levs = []
    for gt, pred in zip(gts, preds):
        levs += [lev(gt.encode("ascii", "ignore").decode(), pred.encode("ascii", "ignore").decode())]
    return np.mean(levs)
