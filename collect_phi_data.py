from misc.tools import build_file_list, balanced_sampling, compute_num_activations

from termcolor import colored, cprint
from numpy.random import choice
import argparse
import h5py
import glob
import pickle
import numpy as np
import time
import sys
import os


def setup_parser():
    """
    Setup the CLI parsing.
    """
    parser = argparse.ArgumentParser(description="Create phi data.")
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
        help='name of (pickled) output file with data')
    parser.add_argument(
        "--data_postfix",       
        metavar='', 
        help="postfix of pickled detection files")
    parser.add_argument(
        "--min_count",          
        metavar='', 
        type=int, 
        default=100,
        help="minimum number of examples per object category to use (default: 100)")
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
        "--interval_file",
        metavar='',
        help = "file with appropriate attribute value intervals")
    parser.add_argument(
        "--verbose",            
        action="store_true", 
        default=False, 
        dest="verbose", 
        help="verbose output")
    return parser


def read_interval_data(file):
    with open(file, 'r') as fid:
        content = fid.readlines()
    content = [x.rstrip() for x in content]

    interval_data = {}
    for cnt, interval_str in enumerate(content):
        a,b = interval_str.split()
        interval_data[cnt] = {
            'data_cnt' : 0,
            'data_feat': None, 
            'data_attr': None,
            'data_oidx': None,
            'interval': (float(a),float(b))}
    return interval_data


def sample_per_object(data, min_count=100):
    unique_vals, counts = np.unique(data, return_counts=True)

    pos = []
    for cnt, val in enumerate(unique_vals):
            if counts[cnt] > min_count:
                idx = np.where(data==val)[0]
                sel = choice(idx, size=min_count, replace=None)
                pos.append(sel)        
    return pos


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = setup_parser().parse_args()

    file_list = build_file_list(
        args.img_list,
        args.img_base,
        args.data_postfix)[args.beg_index:args.end_index]
    assert len(file_list) == args.end_index - args.beg_index

    # compute potential number of total activations
    num = compute_num_activations(file_list)

    # read interval information
    interval_data = read_interval_data(args.interval_file)
    
    for cnt, file_name in enumerate(file_list):

        if args.verbose:
            cprint('[%.5d] File %s' % (cnt, file_name), 'blue')

        with open(file_name,'r') as input:
            data = pickle.load(input)
        if not data['is_valid']:
            continue

        for idx, tmp in enumerate(data['attributes']):
            
            tmp_data_attr = tmp[args.attribute]
            tmp_data_oidx = data['obj_idx'][idx]

            # iterate over all intervals
            for key in interval_data:
                lo = interval_data[key]['interval'][0]
                hi = interval_data[key]['interval'][1]

                if tmp_data_attr >= lo and tmp_data_attr < hi:
                    current_cnt = interval_data[key]['data_cnt']

                    if current_cnt == 0:
                        interval_data[key]['data_feat'] = np.zeros((num, data['CNN_activations'].shape[1]))
                        interval_data[key]['data_attr'] = np.zeros((num,))
                        interval_data[key]['data_oidx'] = np.zeros((num,))
                    
                    
                    interval_data[key]['data_feat'][current_cnt,:] = data['CNN_activations'][idx,:]
                    interval_data[key]['data_attr'][current_cnt] = tmp_data_attr
                    interval_data[key]['data_oidx'][current_cnt] = tmp_data_oidx
                    interval_data[key]['data_cnt'] = current_cnt + 1

    for key in interval_data:

        pos_del = np.arange(
            interval_data[key]['data_cnt'],
            len(interval_data[key]['data_attr']))
        
        data_attr = np.delete(interval_data[key]['data_attr'],pos_del,0)
        data_oidx = np.delete(interval_data[key]['data_oidx'],pos_del,0)
        data_feat = np.delete(interval_data[key]['data_feat'],pos_del,0)

        pos = sample_per_object(data_oidx, min_count=args.min_count)
        if len(pos)>0:
            pos = np.concatenate(pos).ravel()   
            data_attr = data_attr[pos]
            data_oidx = data_oidx[pos]
            data_feat = data_feat[pos,:]
        else:
            data_attr = None
            data_oidx = None
            data_feat = None

        interval_data[key]['data_oidx'] = data_oidx 
        interval_data[key]['data_attr'] = data_attr
        interval_data[key]['data_feat'] = data_feat

    if not args.save is None:
        with open(args.save, 'wb') as fid:
            pickle.dump(interval_data, fid, pickle.HIGHEST_PROTOCOL) 

    if args.verbose:
        for key in interval_data:
            if not interval_data[key]['data_feat'] is None:

                lo = interval_data[key]['interval'][0]
                hi = interval_data[key]['interval'][1]
                N_in_interval = interval_data[key]['data_feat'].shape[0]

                cprint('[%.2f-%.2f]: %.4d activations' % (lo, hi, N_in_interval), 'blue')


if __name__ == "__main__":
    main()