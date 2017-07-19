"""One-shot object recognition experiments with AGA.

Author(s): rkwitt, mdixit, 2017
"""

import sys
sys.path.append("../")

from misc.tools import build_file_list

from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from termcolor import colored, cprint
from scipy.io import loadmat
import numpy as np
import argparse
import glob
import pickle
import time
import sys
import os


def setup_parser():
    parser = argparse.ArgumentParser(description='One-shot object recognition experiments')
    parser.add_argument(
        "--verbose",            
        action="store_true", 
        default=False, 
        dest="verbose", 
        help="enables verbose output")
    parser.add_argument(
        "--img_list",           
        metavar='', 
        help="list with image names (no extension)")
    parser.add_argument(
        "--dim",
        type=int, 
        default=4096, 
        help="dimensionality of features (default: 4096)")
    parser.add_argument(
        "--data_postfix",           
        metavar='', 
        help="postfix of data files (with extension)")
    parser.add_argument(
        "--img_base",           
        metavar='', 
        help="base directory for image data")
    return parser


def collect_data(img_data_files):
    data = {}
    for data_file in img_data_files:
        with open(data_file, 'r') as fid:
            tmp = pickle.load(fid)

        for key in tmp:

            obj_idx = tmp[key]['obj_idx']
            obj_syn = tmp[key]['CNN_activation_syn']
            obj_org = tmp[key]['CNN_activation_org']
            
            if not obj_idx in data:
                data[obj_idx] = []

            data[obj_idx].append((obj_syn, obj_org))
    return data


def select_one_shot(data, args):

    data_trn_syn = None
    data_trn_one = None
    data_tst = None
    data_trn_lab_syn = []
    data_trn_lab_one = []
    data_tst_lab = []

    for obj_id in data:

        c = np.random.choice(
            len(data[obj_id]), 
            size=1, 
            replace=False)[0]   

        syn_data, org_data = data[obj_id][c]

        assert org_data.shape[0] == 1

        data_trn_lab_syn += [obj_id for k in np.arange(syn_data.shape[0]+org_data.shape[0])]
        data_trn_lab_one += [obj_id for k in np.arange(org_data.shape[0])]

        if data_trn_syn is None:
            data_trn_syn = np.vstack((syn_data, org_data))
            data_trn_one = org_data
        else:
            data_trn_syn = np.vstack((data_trn_syn, org_data))
            data_trn_syn = np.vstack((data_trn_syn, syn_data))
            data_trn_one = np.vstack((data_trn_one, org_data))
 
        num = 0
        for j in np.arange(len(data[obj_id])):
            if j == c: 
                continue
            num += data[obj_id][j][0].shape[0]
            num += data[obj_id][j][1].shape[0]

        cnt = 0
        data_tmp = np.zeros((num, args.dim))
        for j in np.arange(len(data[obj_id])):
            if j == c: 
                continue

            i0 = cnt
            i1 = i0+1
            i2 = i1+data[obj_id][j][0].shape[0]
            cnt = i2

            data_tmp[i0,:]  = data[obj_id][j][1]
            data_tmp[i1:i2] = data[obj_id][j][0]

        data_tst_lab += [obj_id for k in np.arange(data_tmp.shape[0])]

        if data_tst is None:
            data_tst = data_tmp
        else:
            data_tst = np.vstack((data_tst, data_tmp))

    return data_trn_syn, data_trn_lab_syn, data_trn_one, data_trn_lab_one, data_tst, data_tst_lab

      
def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = setup_parser().parse_args()

    img_data_files = build_file_list(
        args.img_list,
        args.img_base,
        args.data_postfix
        )

    data = collect_data(img_data_files)

    trn, trn_lab, trn_one, trn_lab_one, tst, tst_lab = select_one_shot(data, args)

    normalizer = Normalizer(norm='l1', copy=True)
    trn = normalizer.fit_transform(trn)
    tst = normalizer.transform(tst)
    
    clf = SVC(kernel='linear', C=1)
    clf.fit(trn, trn_lab) 
    pred = clf.predict(tst)
    print accuracy_score(pred, tst_lab)

    normalizer = Normalizer(norm='l1', copy=True)
    trn = normalizer.fit_transform(trn_one)
    tst = normalizer.transform(tst)
    
    clf = SVC(kernel='linear', C=1)
    clf.fit(trn, trn_lab_one) 
    pred = clf.predict(tst)
    print accuracy_score(pred, tst_lab)




if __name__ == "__main__":
    main()
