"""Utility functions.

Author: rkwitt, mdixit (2017)
"""

from numpy.random import choice
import numpy as np
import pickle
import os


def compute_num_activations(file_list):
    """
    Computes the number of activations available from all
    the data files in a given file list.

    Args:
        file_list (list): A list of file names for data
            files corresponding to images.
    """
    num = 0
    for cnt, file_name in enumerate(file_list):
        with open(file_name,'r') as input:
            data = pickle.load(input)
        if not data['is_valid']:
            continue
        num += len(data['attributes'])
    return num


def compute_SUNRGBD_attributes(img_meta, idx):
    """
    Computes attributes for SUNRGBD objects.

    Note: This could potentially be extended to include
    more attributes.

    Args:
        img_meta (AGAImage): AGAImage object

    Returns:
        attributes (dict): Dictionary with attribute names
            as keys and attribute values as values. Keys are

            depth: Object depth
            angle: Object pose (i.e., angle w.r.t. y axis)
            cname: Object class name
    """

    obj_depth = img_meta.get_gtBB3DCentroid(idx)[1]
    obj_angle = np.arccos(img_meta.get_gtBB3DBasis(idx)[0,0])
    obj_cname = img_meta.get_label(idx)

    return {
        "depth" : obj_depth,
        "angle" : obj_angle,
        "cname" : obj_cname
    }


def balanced_sampling(y):
    """
    Creates a list of indices into an array of object class IDs,
    such that extracting the object class IDs based on the computed
    indices would lead to a balanced sample (i.e., the same number
    of samples per object class would be drawn). The number of
    samples per class is dependend on the least frequently occurring
    class label.

    Args:
        y (np.array): Array of object class indices

    Returns:
        l (list): List of indices into y, such that y(l) is
            balanced in terms of occurrence.
    """
    p = []
    u, counts = np.unique(y,return_counts=True)
    min_count = np.min(counts)
    for val in u:
        idx = np.where(y==val)[0]
        p.append(choice(idx, size=min_count, replace=None))

    return p


def IoU(A,B):
    """
    Computes the Intersection-over-Union union (IoU) between
    two rectangles, given by

    [x1,y1,x2,y2],

    i.e., (x1,y1) - bottom left corner
          (x2,y2) - top right corner

    Args:
        A: np.array (4,)
        B: no.array (4,)

    Returns:
        I: The Intersection-over-Union (IoU) measure
    """
    xx1 = np.max((A[0],B[0]))
    xx2 = np.min((A[2],B[2]))

    yy1 = np.max((A[1],B[1]))
    yy2 = np.min((A[3],B[3]))

    if (xx2 < xx1) or (yy2 < yy1):
        return 0
    else:
        t0 = np.double(A[2]-A[0])
        t1 = np.double(A[3]-A[1])
        t2 = np.double(B[2]-B[0])
        t3 = np.double(B[3]-B[1])

        A1 = t0*t1
        A2 = t2*t3

        I = (xx2-xx1)*(yy2-yy1)
        return I/(A1+A2-I)


def build_file_list(list_file, base=None, postfix=None, remove_suffix=False):
    """
    Reads file names from a file and returns a list of these
    file names with appended postfixes. If a directory base
    name is also given, the file names will be the absolute
    file names.

    Args:
        list_file (string): Filename of list file.
        base (string): Base path to be added.
        postfix (string): Postfix to be added.
        remove_suffix (bool): Remove file suffixes

    Returns:
        filelist (list): List of filenames.
    """
    with open(list_file) as fid:
        content = fid.readlines()
    files = [x.rstrip() for x in content]

    if remove_suffix:
        files = [os.path.splitext(f)[0] for f in files]

    if not postfix is None:
        files = [(x+postfix) for x in files]

    if not base is None:
        files = [os.path.join(base,x) for x in files]

    return files
