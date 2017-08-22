# AGA : Attribute-Guided Augmentation

This repository contains a PyTorch implementation of

```
@inproceedings{Dixit17a,	
	author    = {M.~Dixit and R.~Kwitt and M.~Niethammer and N.~Vasconcelos},
	title     = {AGA : Attribute-Guided Augmentation},
	booktitle = {CVPR},
	year      = 2017}
```

**Disclaimer**: This documentation is still in development! 

## Requirements

### Software

To reproduce our results on SUNRGBD from scratch, we need to following packages

- [Fast-RCNN](https://github.com/rbgirshick/fast-rcnn.git)
- [SelectiveSearch](https://github.com/sergeyk/selective_search_ijcv_with_python0)
- [PyTorch](http://pytorch.org/)
- [LibLinear](https://github.com/ninjin/liblinear)
- [scikit-learn](http://scikit-learn.org/stable)

### Data

We provide a couple of files that have already prepared SUNRGBD data and a
Fast-RCNN model, finetuned to a selection of SUNRGBD object classes. These
files can we downloaded via

- [SUNRGBD images](https://drive.google.com/file/d/0BxHF82gaPzgScVJ5LXVCWkRsaE0/view?usp=sharing)
- [Finetuned VGG16 model](https://drive.google.com/file/d/0BxHF82gaPzgSY2gySFpQU0hJOVU/view?usp=sharing)

We will assume, from now own, that the images are unpacked into `/data/images/`. The 
finetuned model has to be put into the `<PATH_TO_FAST_RCNN_DIR>/data/fast_rcnn_models`
folder.

## Running Fast-RCNN

Say, you have unpacked the original SUNRGBD images to `/data/images`, i.e., the 
directory should contain `image_00001.jpg` to `image_10335.jpg`. We also assume
that you have run *SelectiveSearch* on each image (in MATLAB) and stored the 
bounding box proposals for each image as `image_00001_ss_boxes.mat` in the same
folder (i.e., `/data/images`).

First, we create a file which contains all image filenames without extension.

```bash
cd /data/images
find . -name '*.jpg' -exec basename {} .jpg \; > image_list.txt
```

Next, we run the Fast-RCNN detector with

```bash
cd <PATH_TO_AGA_DIR>
cp misc/object_features.py <PATH_TO_FAST_RCNN_DIR>/tools
cd <PATH_TO_FAST_RCNN_DIR>
python tools/object_features.py
```

This generates (in `/data/images`), for each image, a file `<IMAGE_NAME>_bbox_features.mat` file
which contains Fast-RCNN (FC7) features with detection scores for each object class that we 
finetuned on.

## Collecting data for AGA training

```bash
cd <AGA_BASE_DIR>
python collect_train.py \ 
	--img_meta datasets/SUNRGBD/SUNRGBD_meta.pkl \
	--img_list /data/images/allimgs.txt \
	--img_base /data/images \
	--bbox_postfix _ss_boxes.mat \
	--data_postfix _bbox_features.mat \
	--outfile_postfix _fc7.pkl \
	--object_class_file datasets/SUNRGBD/SUNRGBD_objects.pkl
```

This creates, for each image, a Python pickle file `<IMAGE_NAME>_fc7.pkl` that contains
Fast-RCNN FC7 activations for all bounding boxes that overlap with the ground truth by
IoU > 0.7 and detection scores for object classes > 0.5. Detections for `__background__`
and `others` are also excluded. Further, these files contain annotations (per remaining
bounding box) for *depth* and *pose*.

## Training object attribute strength predictors

tbd.

## Training the synthesis function

tbd.

## Synthesis via AGA

tbd.

## Experiments

tbd.

### One-shot object recognition

tbd.

















