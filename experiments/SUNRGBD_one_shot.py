"""One-shot object recognition experiments with AGA.

Author(s): rkwitt, mdixit, 2017
"""

import sys
sys.path.append("../")

from misc.tools import build_file_list, balanced_sampling

from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from termcolor import colored, cprint
from scipy.io import loadmat
from scipy import stats
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
        "--runs",
        type=int,
        default=10,
        help="number of evaluation runs (default: 10)")
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

    # iterate over all data files
    for data_file in img_data_files:
        with open(data_file, 'r') as fid:
            tmp = pickle.load(fid)

        # iterate over all available detections for that image
        for det_idx in tmp:

            obj_idx = tmp[det_idx]['obj_idx']            # Object ID
            obj_syn = tmp[det_idx]['CNN_activation_syn'] # AGA-syn. feature(s)
            obj_org = tmp[det_idx]['CNN_activation_org'] # Original feature

            if not obj_idx in data:
                data[obj_idx] = []

            # store AGA-syn. + original features as a list of tuples per
            # object class.
            data[obj_idx].append((obj_syn, obj_org))
    return data


def select_one_shot(data, args):

    data_trn_syn = None     # AGA-syn. training data
    data_trn_one = None     # Original single-instance data
    data_tst = None         # Testing data

    data_trn_syn_lab = []   # AGA-syn. training labels
    data_trn_one_lab = []   # Original single-instance labels
    data_tst_lab = []       # Testing labels

    # iterate over all objects classes
    for obj_id in data:

        # choose a sample from object class at random
        c = np.random.choice(
            len(data[obj_id]),
            size=1,
            replace=False)[0]

        # extract AGA-syn. + original single-instance features
        syn_data, org_data = data[obj_id][c]
        assert org_data.shape[0] == 1
        assert syn_data.shape[0]  > 0

        # construct label lists
        data_trn_syn_lab += [obj_id for k in np.arange(syn_data.shape[0]+org_data.shape[0])]
        data_trn_one_lab += [obj_id for k in np.arange(org_data.shape[0])]

        # build-up training data
        if data_trn_syn is None:
            data_trn_syn = np.vstack((syn_data, org_data)) # stack original + AGA-syn. features
            data_trn_one = org_data                        # only add original features
        else:
            data_trn_syn = np.vstack((data_trn_syn, org_data)) # add original features
            data_trn_syn = np.vstack((data_trn_syn, syn_data)) # add AGA-syn. features
            data_trn_one = np.vstack((data_trn_one, org_data)) # add original features
        assert data_trn_syn.shape[0] == len(data_trn_syn_lab)

        # next, we construct the object-class specific testing data
        num = 0
        for j in np.arange(len(data[obj_id])):
            if j == c:
                continue
            _,org_data = data[obj_id][j]
            num += org_data.shape[0]
        assert (len(data[obj_id])-1)==num

        cnt = 0
        data_tmp = np.zeros((num, args.dim))
        for j in np.arange(len(data[obj_id])):
            if j == c:
                continue
            _,org_data = data[obj_id][j]
            assert org_data.shape[0] == 1
            data_tmp[cnt,:]  = org_data
            cnt += 1

        if data_tst is None:
            data_tst = data_tmp
        else:
            data_tst = np.vstack((data_tst, data_tmp))

        data_tst_lab += [obj_id for k in np.arange(data_tmp.shape[0])]

    ret_data = {
        "data_trn_syn":     data_trn_syn,
        "data_trn_one":     data_trn_one,
        "data_trn_syn_lab": data_trn_syn_lab,
        "data_trn_one_lab": data_trn_one_lab,
        "data_tst":         data_tst,
        "data_tst_lab":     data_tst_lab}
    return ret_data


def eval(Xtrn, Xtrn_lab, Xtst, Xtst_lab):

    # normalizer = MinMaxScaler(copy=True)
    # norm_Xtrn = normalizer.fit_transform(Xtrn)
    # norm_Xtst = normalizer.transform(Xtst)

    normalizer = Normalizer(
       norm='l1',
       copy=True)
    norm_Xtrn = normalizer.transform(Xtrn)
    norm_Xtst = normalizer.transform(Xtst)

    # train a linear C-SVM
    clf = LinearSVC(C=1)
    clf.fit(norm_Xtrn, Xtrn_lab)

    p = np.concatenate(balanced_sampling(Xtst_lab)).ravel()
    predictions = clf.predict(norm_Xtst[p,:])

    return {
        'norm_Xtrn': norm_Xtrn,
        'norm_Xtst': norm_Xtst,
        'accuracy_score': accuracy_score(np.asarray(Xtst_lab)[p], predictions)}

def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = setup_parser().parse_args()

    img_data_files = build_file_list(
        args.img_list,
        args.img_base,
        args.data_postfix
        )

    data_source = collect_data(img_data_files)

    if args.verbose:
        for obj_id in data_source:
            cprint('Object: {} - {} detections'.format(
                obj_id, len(data_source[obj_id])), 'blue')

    Z = np.zeros((args.runs, 2))
    for run_id in np.arange(args.runs):

        data = select_one_shot(data_source, args)

        tmp_result_one = eval(
            data['data_trn_one'],
            data['data_trn_one_lab'],
            data['data_tst'],
            data['data_tst_lab'])

        tmp_result_syn = eval(
            data['data_trn_syn'],
            data['data_trn_syn_lab'],
            data['data_tst'],
            data['data_tst_lab'])

        Z[run_id,0] = tmp_result_one['accuracy_score']
        Z[run_id,1] = tmp_result_syn['accuracy_score']

        if args.verbose:
            cprint('[{}] {:.2f} - {:.2f}'.format(
                run_id,
                tmp_result_one['accuracy_score'],
                tmp_result_syn['accuracy_score']), 'blue')

    print Z.mean(axis=0)
    print stats.wilcoxon(Z[:,0],Z[:,1])



if __name__ == "__main__":
    main()
