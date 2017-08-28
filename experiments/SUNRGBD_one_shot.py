"""One-shot object recognition experiments with AGA.

NOTE: this code has been replaced by SUNRGBD_few_shot.py and will be removed soon!

Author(s): rkwitt, mdixit, 2017
"""

import sys
sys.path.append("../")
sys.path.append("liblinear-2.11/python")

import liblinear
import liblinearutil
from misc.tools import build_file_list, balanced_sampling

import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from termcolor import colored, cprint
from scipy.io import loadmat, savemat
from scipy import stats
import numpy as np
import argparse
import glob
import pickle
import time
import sys
import os


class ResultStatistics:

    def __init__(self):
        self._container = {}

    def add_result(self, tag, val):
        if not tag in self._container:
            self._container[tag] = []
        self._container[tag].append(val)


    def print_current(self, keys=None):
        out_str = ''
        if keys is None:
            for key in self._container:
                out_str += '[{}] {}: {:.2f} | '.format(
                    str(len(self._container[key])).zfill(5),
                    key,
                    self._container[key][-1])
        else:
            for key in keys:
                out_str += '[{}] {}: {:.2f} | '.format(
                    str(len(self._container[key])).zfill(5),
                    key,
                    self._container[key][-1])
        cprint(out_str, 'blue')


    def print_summary(self, keys=None):
        out_str = ''
        if keys is None:
            for key in self._container:
                avg = np.array([self._container[key]]).mean()
                out_str += '{}: {:.2f} | '.format(key, avg)
        else:
            for key in keys:
                avg = np.array([self._container[key]]).mean()
                out_str += '{}: {:.2f} | '.format(key, avg)
        cprint(out_str,'red')


def setup_parser():
    parser = argparse.ArgumentParser(description='One-shot object recognition experiments')
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        dest="verbose",
        help="enables verbose output")
    parser.add_argument(
        "--omit_original",
        action="store_true",
        default=False,
        dest="omit_original",
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
        print data_file

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

    # **DEBUG** - for comparision to MATLAB version
    debug_indices = [44,
        110,
        47,
        65,
        83,
        23,
        117,
        97,
        632,
        128]
    debug_indices = [u-1 for u in debug_indices]

    # iterate over all objects classes
    for m, obj_id in enumerate(data):

        #choose a sample from object class at random
        c = np.random.choice(
               len(data[obj_id]),
           size=1,
           replace=False)[0]

        # **DEBUG**
        # c = debug_indices[m]

        # extract AGA-syn. + original single-instance features
        syn_data, org_data = data[obj_id][c]

        assert org_data.shape[0] == 1
        assert syn_data.shape[0]  > 0

        # construct label lists
        if args.omit_original:
            data_trn_syn_lab += [obj_id for k in np.arange(syn_data.shape[0])]
        else:
            data_trn_syn_lab += [obj_id for k in np.arange(syn_data.shape[0]+org_data.shape[0])]
        data_trn_one_lab += [obj_id for k in np.arange(org_data.shape[0])]

        # build-up training data
        if data_trn_syn is None:
            assert data_trn_one is None
            assert data_trn_syn is None

            if args.omit_original:
                data_trn_syn = syn_data
            else:
                data_trn_syn = np.vstack((syn_data, org_data))      # stack original + AGA-syn. features
            data_trn_one = org_data                                 # only add original features
        else:
            if args.omit_original:
                data_trn_syn = np.vstack((data_trn_syn, syn_data))  # add AGA-syn. features
                data_trn_one = np.vstack((data_trn_one, org_data))  # add original features
            else:
                data_trn_syn = np.vstack((data_trn_syn, org_data))  # add original features
                data_trn_syn = np.vstack((data_trn_syn, syn_data))  # add AGA-syn. features
                data_trn_one = np.vstack((data_trn_one, org_data))  # add original features
        assert data_trn_syn.shape[0] == len(data_trn_syn_lab)

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


def eval_SVM(X, y, Xhat, yhat):
    # create classification problem
    problem = liblinearutil.problem(y,X)

    # set SVM parameters
    svm_param = liblinearutil.parameter('-s 3 -c 10 -q -B 1')

    # train SVM
    model = liblinearutil.train(problem, svm_param)

    # predict and evaluate
    p_label, p_acc, p_val = liblinearutil.predict(yhat, Xhat, model, '-q')

    # compute accuracy
    acc, mse, scc = liblinearutil.evaluations(yhat, p_label)
    return acc


def eval_NN1(X,y,Xhat,yhat):
    # create 1-NN classifier
    neigh = KNeighborsClassifier(n_neighbors=1)

    # train :)
    neigh.fit(X, y)

    # compute accuracy
    acc = accuracy_score(yhat, neigh.predict(Xhat))
    return acc*100.0


def eval(Xtrn, Xtrn_lab, Xtst, Xtst_lab):

    # guarantee valid activations
    Xtrn = Xtrn.clip(min=0)
    Xtst = Xtst.clip(min=0)

    # L1 normalization
    normalizer = Normalizer(
       norm='l1',
       copy=True)
    norm_Xtrn = normalizer.fit_transform(Xtrn)
    norm_Xtst = normalizer.transform(Xtst)

    # possibly cleanup numerics
    norm_Xtrn[np.abs(norm_Xtrn) < 1e-9] = 0
    norm_Xtst[np.abs(norm_Xtst) < 1e-9] = 0

    # create sparse matrix (as many entries are 0 anyways)
    norm_Xtrn_sparse = scipy.sparse.csr_matrix(norm_Xtrn)
    norm_Xtst_sparse = scipy.sparse.csr_matrix(norm_Xtst)

    svm_acc = eval_SVM(
        norm_Xtrn_sparse,
        Xtrn_lab,
        norm_Xtst_sparse,
        Xtst_lab)
    nn1_acc = eval_NN1(
        norm_Xtrn_sparse,
        Xtrn_lab,
        norm_Xtst_sparse,
        Xtst_lab)

    return {'SVM' : svm_acc, '1NN' : nn1_acc}


def main(argv=None):
    if argv is None:
        argv = sys.argv

    np.random.seed(seed=1234)
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

    stats = ResultStatistics()
    for run_id in np.arange(args.runs):

        data = select_one_shot(data_source, args)

        tmp_result_one = eval(
            data['data_trn_one'],
            data['data_trn_one_lab'],
            data['data_tst'],
            data['data_tst_lab'])

        stats.add_result('SVM (w/o AGA)', tmp_result_one['SVM'])
        stats.add_result('1NN (w/o AGA)', tmp_result_one['1NN'])

        tmp_result_syn = eval(
            data['data_trn_syn'],
            data['data_trn_syn_lab'],
            data['data_tst'],
            data['data_tst_lab'])

        stats.add_result('SVM (w AGA)', tmp_result_syn['SVM'])
        stats.add_result('1NN (w AGA)', tmp_result_syn['1NN'])

        stats.print_current([
            'SVM (w/o AGA)',
            'SVM (w AGA)',
            '1NN (w/o AGA)',
            '1NN (w AGA)'])

    stats.print_summary([
        'SVM (w/o AGA)',
        'SVM (w AGA)',
        '1NN (w/o AGA)',
        '1NN (w AGA)'])


if __name__ == "__main__":
    main()
