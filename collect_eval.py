"""
Collect detections for evaluation of AGA.

Author: rkwitt, mdixit (2017)
"""

from misc.AGAImage import AGAImage
from misc.tools import build_file_list

from termcolor import colored, cprint
from scipy.io import loadmat
import argparse
import glob
import pickle
import numpy as np
import time
import sys
import os


def setup_parser():
    parser = argparse.ArgumentParser(description="Collect detection/activation data")

    parser.add_argument(
        "--img_meta",
        metavar='',
        help="pickle file with image meta information")
    parser.add_argument(
        "--img_list",
        metavar='',
        help="file with image filenames (no extension)")
    parser.add_argument(
        "--img_base",
        metavar='',
        help="base directory of image files")
    parser.add_argument(
        "--data_postfix",
        metavar='',
        help="postfix to identify bounding box files")
    parser.add_argument(
        "--label_postfix",
        metavar='',
        help="postfix to identify label files")
    parser.add_argument(
        "--outfile_postfix",
        metavar='',
        help="postfix to add to output files (including extension)")
    parser.add_argument(
        "--object_class_file",
        metavar='',
        help='pickeled file with a list of object names')
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        dest="verbose",
        help="Verbose output")
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = setup_parser().parse_args()

    with open(args.object_class_file, 'r') as input:
        object_class_names = pickle.load(input)

    # list of image file names
    img_files = build_file_list(args.img_list)

    # list of feature file names
    img_feat_files = build_file_list(
        args.img_list,
        base=args.img_base,
        postfix=args.data_postfix)

    # list of label files
    img_label_files = build_file_list(
        args.img_list,
        base=args.img_base,
        postfix=args.label_postfix)

    for cnt,_ in enumerate(img_files):

        detections = {
            'is_valid'       : True,    # is detection entry valid
            'CNN_scores'     : None,    # detection scores / object class
            'CNN_activations': None,    # CNN activations / bounding box
            'attributes'     : None,    # attributes associated with bounding box
            'det_idx'        : None,    # index of kept detection box
            'det_box'        : None,    # coordinates of detection box
            'gtb_idx'        : None,    # index of most-overlapping ground truth box
            'gtb_box'        : None,    # coordinates of most-overlapping ground truth box
            'obj_cls'        : None,    # object classes
            'obj_idx'        : None,    # indices of object classes
        }

        t0 = time.time()
        data = loadmat(img_feat_files[cnt])['CNN_feature']
        labels = list(loadmat(img_label_files[cnt])['labels'].flatten()-1)

        detections['CNN_activations'] = data
        detections['obj_idx'] = list(labels)
        detections['obj_cls'] = [object_class_names[l] for l in labels]
        ela = time.time() - t0

        if args.verbose:
            cprint('Done with {} [{}, in {:.4f} s]'.format(
                img_files[cnt],
                data.shape[0],
                ela), 'blue')

        outfile = os.path.join(
            args.img_base,
            img_files[cnt] + args.outfile_postfix)
        with open(outfile, 'wb') as fid:
            pickle.dump(detections, fid, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
