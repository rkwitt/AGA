"""Collect training data for attribute strength predictor (rho) training.

Author: rkwitt, mdixit (2017)
"""

from misc.tools import build_file_list, balanced_sampling, compute_num_activations

from termcolor import colored, cprint
import argparse
import h5py
import glob
import pickle
import numpy as np
import time
import sys
import os


def setup_parser():
    parser = argparse.ArgumentParser(description="Collect data for training of rho.")
    parser.add_argument(
        "--img_list",           
        metavar='', 
        help="file with image filenames (no extension)")
    parser.add_argument(
        "--img_base",           
        metavar='', 
        help="base directory of image files")
    parser.add_argument(
        "--attribute",          
        metavar='', 
        help="attribute name")
    parser.add_argument(
        "--save", 
        metavar='', 
        help='name of the (pickle) output file with balanced data')
    parser.add_argument(
        "--data_postfix",       
        metavar='', 
        help="postfix of detection files")
    parser.add_argument(
        "--beg_index",          
        metavar='', 
        type=int, 
        help="start index from which to extract data")
    parser.add_argument(
        "--end_index",          
        metavar='', 
        type=int, 
        help="stop index from which to extract data")
    parser.add_argument(
        "--verbose",            
        action="store_true", 
        default=False, 
        dest="verbose", 
        help="berbose output")
    parser.add_argument(
        '--no_sampling',            
        action='store_true', 
        default=False, 
        help='disables balanced sampling (default: False)')
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = setup_parser().parse_args()
    args.sampling = not args.no_sampling

    file_list = build_file_list(
        args.img_list,
        args.img_base,
        args.data_postfix)[args.beg_index:args.end_index]

    assert len(file_list) == args.end_index - args.beg_index

    data_init = False
    data_feat = None
    data_oidx = None
    data_attr = None

    num = compute_num_activations(file_list)
    
    feat_cnt = 0
    for cnt, file_name in enumerate(file_list):

        if args.verbose:
            cprint('[%.5d] File %s' % (cnt, file_name), 'blue')

        with open(file_name,'r') as input:
            data = pickle.load(input)
        if not data['is_valid']:
            continue

        if not data_init:
            data_feat = np.zeros((num, data['CNN_activations'].shape[1]))
            data_oidx = np.zeros((num,))
            data_attr = np.zeros((num,))
            data_init = True
        
        N_current = data['CNN_activations'].shape[0]
        data_feat[feat_cnt:feat_cnt+N_current,:] = data['CNN_activations']
        data_oidx[feat_cnt:feat_cnt+N_current] = data['obj_idx']
        for j, tmp in enumerate(data['attributes']):
            data_attr[feat_cnt+j] = tmp[args.attribute]
        feat_cnt += N_current

    if args.verbose:
        cprint('Total amount of data (%d x %d)' % data_feat.shape, 'blue')

    if args.sampling:      
        sample_arr = balanced_sampling(data_oidx)
        sample_idx = np.concatenate(sample_arr).ravel()

        data = {
            'data_feat' : data_feat[sample_idx,:],
            'data_oidx' : data_oidx[sample_idx],
            'data_attr' : data_attr[sample_idx]}
    else:
        data = {
            'data_feat' : data_feat,
            'data_oidx' : data_oidx,
            'data_attr' : data_attr}

    if not args.save is None:
        with open(args.save, 'wb') as fid:
            pickle.dump(data, fid, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()