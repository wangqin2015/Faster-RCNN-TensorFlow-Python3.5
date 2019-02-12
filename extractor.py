#!/usr/bin/env python


"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from math import ceil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
# from lib.nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from lib.model.bbox_transform import clip_boxes, bbox_transform_inv
from lib.utils.blob import im_list_to_blob

classes = ('__background__',
               'apple', 'tree', 'yellow', 'blue', 'white', 'histogram')

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.FLAGS2["pixel_means"]

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.FLAGS2["test_scales"]:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.FLAGS.test_max_size:
      im_scale = float(cfg.FLAGS.test_max_size) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)



def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""
    for i in range(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    # seems to have height, width, and image scales
    # still not sure about the scale, maybe full image it is 1.
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    _, scores, bbox_pred, rois, feats= net.test_image(sess, blobs['data'], blobs['im_info'])

    boxes = rois[:, 1:5] / im_scales[0]

    if cfg.FLAGS.test_bbox_reg:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes, feats


def get_boxes(sess, net, img, all_feats, all_boxes):
    boxes = {}
    for cls_ind, cls in enumerate(classes):
        if cls == '__background__':
            continue
        dets = all_boxes[cls_ind]
        feats = all_feats[cls_ind]
        if dets == []:
            continue
        for k in range(dets.shape[0]):
            det = dets[k]
            box = {}
            box['lt'] = [int(det[0]), int(det[1])]
            box['rb'] = [int(det[2]), int(det[3])]
            box['f'] = feats[k]
            box['cl'] = cls
            box['score'] = det[4]
            box_width = box['rb'][0] - box['lt'][0]
            box_height = box['rb'][1] - box['lt'][1]
            min_size = min(box_width, box_height)
            if min_size > 10:
                if not cls in boxes:
                    boxes[cls] = []
                class_boxes = boxes[cls]
                # boxes.append(box)
                class_boxes.append(box)
    return boxes


def read_img(path):
    " Read image from file "
    im = cv2.imread(path).astype(np.float32, copy=True)
    return im


def extract_regions_and_feats(sess, net, im, max_per_image=10, max_per_class=3, thresh=0.1):
    """Extract regions and feature corresponding to the each box
    """
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    # timecrs
    _t = {'im_detect': Timer(), 'misc': Timer()}
    if type(im) == str:
        im = cv2.imread(im)

    _t['im_detect'].tic()
    scores, boxes, feats = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    all_boxes = [[] for _ in range(cfg.FLAGS.class_num)]
    all_feats = [[] for _ in range(cfg.FLAGS.class_num)]
    # skip j = 0, because it's the background class
    for j in range(1, cfg.FLAGS.class_num):
        image_thresh = np.sort(scores[:, j])[-max_per_image]
        th = thresh if thresh > image_thresh else image_thresh
        inds = np.where(scores[:, j] > th)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.FLAGS.rpn_test_nms_thresh)
        cls_dets = cls_dets[keep, :]
        feats_part = feats[keep, :]
        all_boxes[j] = cls_dets
        all_feats[j] = feats_part

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][:, -1]
                                  for j in range(1, cfg.FLAGS.class_num)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, 81):
                keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                all_boxes[j] = all_boxes[j][keep, :]
                all_feats[j] = all_feats[j][keep, :]
    _t['misc'].toc()

    # print 'im_detect in {:.3f}s {:.3f}s' \
    # .format(_t['im_detect'].average_time,
    # _t['misc'].average_time)

    boxes = get_boxes(sess, net, im, all_feats, all_boxes)
    return boxes


def extract_imfea(sess, net, img):
    "Extract feature for image"
    # resized image first
    resized_im = cv2.resize(img, (224, 224))
    resized_im=resized_im.astype(np.float)
    resized_im -= cfg.FLAGS2["pixel_means"]
    fea = net.extract_fc7(sess, [resized_im])
    return np.squeeze(fea)


def binarize_fea(fea, thresh=0.1):
    "Binarize and pack feature vector"
    binary_vec = np.where(fea >= thresh, 1, 0)
    f_len = binary_vec.shape[0]
    if f_len % 32 != 0:
        new_size = int(ceil(f_len / 32.) * 32)
        num_pad = new_size - f_len
        binary_vec = np.pad(binary_vec, (num_pad, 0), 'constant')

    return np.packbits(binary_vec).view('uint32')


def get_tags(sess,net,img):
    boxes = extract_regions_and_feats(sess, net, img)
    out = {}
    for cl, bb in boxes.items():
        # print(cl,bb)
        best_score = max([b['score'] for b in bb])
        out[cl] = float(best_score)
    # print(out)
    return out



if __name__ == '__main__':
    # model_path = '.\data\models\\vgg16_faster_rcnn_iter_20000.ckpt'

    # set config
    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth = True
    # sess = tf.Session(config=tfconfig)

    # model_path = '.\\data\\models\\voc2007_2w\\vgg16_faster_rcnn_iter_20000.ckpt'
    model_path = '.\\data\\models\\pikaqiu\\vgg16_faster_rcnn_iter_10.ckpt'
    # model_path = '.\\data\\models\\vgg16_faster_rcnn_iter_490000.ckpt'
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    net = vgg16(batch_size=1)
    num_classes = cfg.FLAGS.class_num

    anchors = [8, 16, 32]
    net.create_architecture(sess, "TEST", num_classes, tag='default', anchor_scales=anchors)
    # net.fix_variables(sess,model_path)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # img = '.\\data\\demo\\004545.jpg'
    img = '.\\data\\demo\\pikachu_1768.jpg'
    if type(img) == str:
        img = cv2.imread(img)


    # extractor = Extractor(model_path, sess=sess)
    # extractor.get_tags(img)
    # fea = extractor.extract_imfea(img)
    # bin_fea = extractor.binarize_fea(fea)
    # print(bin_fea.shape)
    # sess.close()

    tags=get_tags(sess,net,img)
    print(tags)
    # fea=extract_imfea(sess, net, img)
    # print(fea.shape,fea)
    # bin_fea=binarize_fea(fea)
    # print(bin_fea)

