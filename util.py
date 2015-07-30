#!/usr/bin/env python

import numpy as np
import math

def bearingFromVector(dir_unit_vector):
    dx, dy = dir_unit_vector
    return np.degrees(np.arctan(-1*dx/dy))

def circleCorners(center, radius):
    x, y = center
    tl_x = x - radius
    tl_y = y - radius
    br_x = x + radius
    br_y = y + radius
    return [tl_x, tl_y, br_x, br_y]

def average(li):
    li = [x for x in li if x is not None]
    return sum(li) / len(li)

def toScreen(arr, resolution=30):
    return [j * resolution for j in arr]

def distance(p1, p2):
    return abs(math.sqrt(sum([pow(x2 - x1, 2) for x1, x2 in zip(p1, p2)])))

def inside(pos, bound_pos, bound_shape):
    within_dimensions = 0
    for x, rx, rs in zip(pos, bound_pos, bound_shape):
        if x >= rx and x <= (rx+rs):
            within_dimensions += 1
        else:
            continue
    if within_dimensions == len(bound_shape):
        return True
    else:
        return False

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])