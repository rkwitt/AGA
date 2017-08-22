#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.util import im_extract_feat
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

# Directory where all prepared SUNRGBD data is stored
SOURCE_DIR = '/scratch2/rkwitt/data/SUNRGBD/images2/'

CLASSES = ('__background__',
       'others', 'bathtub', 'bed', 'bookshelf', 'box', 'chair',
       'counter', 'desk', 'door', 'dresser', 'garbage bin',
       'lamp', 'monitor', 'night stand', 'pillow', 'sink',
       'sofa', 'table', 'tv', 'toilet')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000_sunrgbd.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo_feat(net, image_name, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(SOURCE_DIR, image_name + '_ss_boxes.mat')
    obj_proposals = sio.loadmat(box_file)['boxes']

    # Load the demo image
    im_file = os.path.join(SOURCE_DIR, image_name + '.jpg')
    im = cv2.imread(im_file)

    feat_file = os.path.join(SOURCE_DIR, image_name + '_bbox_features.mat')
    if not os.path.isfile( feat_file ):

        # Detect all object classes and regress object bounds
        timer = Timer()
        
        timer.tic()
        scores, feats, boxes = im_extract_feat(net, im, obj_proposals, 'fc7')
        timer.toc()
        
        print ('Detection took {:.3f}s for '
            '{:d} object proposals').format(timer.total_time, boxes.shape[0])
        sio.savemat(feat_file, {'CNN_feature': feats, 'CNN_scores': scores})
        print "saving feature file {} size {}".format(feat_file, feats.shape)

    else:
        
        print "file ", feat_file, " exists -> skipping"
        

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR,
        'models',
        NETS[args.demo_net][0],
        'test.prototxt')

    caffemodel = os.path.join(cfg.ROOT_DIR,
        'data',
        'fast_rcnn_models',
        NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    imglist = None
    with open(os.path.join(SOURCE_DIR, 'allimgs.txt')) as fid:
        imglist = fid.readlines()
    imglist = [x.rstrip() for x in imglist]

    for i, img in enumerate(imglist):
        print img

        # im_file = os.path.join(SOURCE_DIR, img + '.jpg')
        # im = cv2.imread(im_file)
        # print im.shape

        # box_file = os.path.join(SOURCE_DIR, img + '_ss_boxes.mat')
        # obj_proposals = sio.loadmat(box_file)['boxes']
        # print obj_proposals

        # im = im[:, :, (2, 1, 0)]
        # fig, ax = plt.subplots(figsize=(12, 12))
        # ax.imshow(im, aspect='equal')

        # for n in np.arange(obj_proposals.shape[0]):
        #     bbox = obj_proposals[n,:]

        #     ax.add_patch(   
        #         plt.Rectangle((bbox[0], bbox[1]),
        #             bbox[2] - bbox[0],
        #             bbox[3] - bbox[1], fill=False,
        #             edgecolor='red', linewidth=3.5))


        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        
        # raw_input()
        demo_feat(net, imglist[i], CLASSES)







