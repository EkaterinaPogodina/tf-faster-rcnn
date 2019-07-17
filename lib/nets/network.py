# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from model.config import cfg


class Network(object):
  def __init__(self):
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._variables_to_fix = {}

  def _add_gt_image(self):
    # add back mean
    image = self._image + cfg.PIXEL_MEANS
    # BGR to RGB (opencv uses BGR)
    resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
    self._gt_image = tf.reverse(resized, axis=[-1])

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_top_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                     self._feat_stride, self._anchors, self._num_anchors],
                                    [tf.float32, tf.float32], name="proposal_top")
      rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
      rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    #print(session.run(rpn_bbox_pred))
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors],
                                    [tf.float32, tf.float32], name="proposal")
      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bounding boxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      pre_pool_size = cfg.POOLING_SIZE * 2
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  # def _anchor_target_layer(self, rpn_cls_score, name, postfix=''):
  #   with tf.variable_scope(name) as scope:
  #     rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
  #       anchor_target_layer,
  #       [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
  #       [tf.float32, tf.float32, tf.float32, tf.float32],
  #       name="anchor_target")
  #
  #     rpn_labels.set_shape([1, 1, None, None])
  #     rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
  #     rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
  #     rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])
  #
  #     rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
  #     self._anchor_targets['rpn_labels' + postfix] = rpn_labels
  #     self._anchor_targets['rpn_bbox_targets' + postfix] = rpn_bbox_targets
  #     self._anchor_targets['rpn_bbox_inside_weights' + postfix] = rpn_bbox_inside_weights
  #     self._anchor_targets['rpn_bbox_outside_weights' + postfix] = rpn_bbox_outside_weights
  #
  #   return rpn_labels

  def _anchor_target_layer(self, rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
        anchor_target_layer,
        [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
        [tf.float32, tf.float32, tf.float32, tf.float32],
        name="anchor_target")

      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

    return rpn_labels

  # def _proposal_target_layer(self, rois, roi_scores, name, postfix=''):
  #   with tf.variable_scope(name + postfix) as scope:
  #     # if postfix:
  #     gt_boxes = self._gt_boxes
  #     # else:
  #     #   gt_boxes = self._gt_boxes_prev
  #
  #     # rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, tracks = tf.py_func(
  #     #   proposal_target_layer,
  #     #   [rois, roi_scores, gt_boxes, self._num_classes],
  #     #   [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
  #     #   name="proposal_target")
  #
  #     rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
  #       proposal_target_layer,
  #       [rois, roi_scores, gt_boxes, self._num_classes],
  #       [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
  #       name="proposal_target")
  #
  #     rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
  #     roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
  #     labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
  #     # tracks.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
  #     bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
  #     bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
  #     bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
  #
  #     self._proposal_targets['rois' + postfix] = rois
  #     self._proposal_targets['labels' + postfix] = tf.to_int32(labels, name="to_int32")
  #     self._proposal_targets['bbox_targets' + postfix] = bbox_targets
  #     self._proposal_targets['bbox_inside_weights' + postfix] = bbox_inside_weights
  #     self._proposal_targets['bbox_outside_weights' + postfix] = bbox_outside_weights
  #     # self._proposal_targets['tracks' + postfix] = tracks
  #
  #     return rois, roi_scores

  def _proposal_target_layer(self, rois, roi_scores, name):
    with tf.variable_scope(name + postfix) as scope:
      # if postfix:
      gt_boxes = self._gt_boxes
      # else:
      #   gt_boxes = self._gt_boxes_prev

      # rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, tracks = tf.py_func(
      #   proposal_target_layer,
      #   [rois, roi_scores, gt_boxes, self._num_classes],
      #   [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
      #   name="proposal_target")

      rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
        proposal_target_layer,
        [rois, roi_scores, gt_boxes, self._num_classes],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        name="proposal_target")

      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
      roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
      labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      # tracks.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

      self._proposal_targets['rois'] = rois
      self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets['bbox_targets'] = bbox_targets
      self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
      # self._proposal_targets['tracks' + postfix] = tracks

      return rois, roi_scores

  def _anchor_component(self, prev=False):

    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right
      height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
      anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                          [height, width,
                                           self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                          [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      # if not prev:
      self._anchors = anchors
      self._anchor_length = anchor_length

      # else:
      #   self._prev_anchors = anchors
      #   self._prev_anchor_length = anchor_length

  def _build_network(self, is_training=True):
    # select initializers
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    net_conv = self._image_to_head(is_training)
    with tf.variable_scope(self._scope, self._scope):
      self._anchor_component()
      rois, rpn = self._region_proposal(net_conv, is_training, initializer)
      pool5 = self._crop_pool_layer(net_conv, rois, "pool5")

    # with tf.variable_scope(self._prev_scope, self._prev_scope):
    #   self._anchor_component(prev=True)
    #   prev_rois, prev_rpn = self._region_proposal(net_conv, is_training, initializer, postfix='_prev')
    #   prev_pool5 = self._crop_pool_layer(net_conv, rois, "pool5")

    fc7 = self._head_to_tail(pool5, is_training)
    # fc7_2 = self._head_to_tail(prev_pool5, is_training, prev=True)
    with tf.variable_scope(self._scope, self._scope):
      # region classification
      cls_prob, bbox_pred = self._region_classification(fc7, is_training, 
                                                        initializer, initializer_bbox)

    # with tf.variable_scope(self._prev_scope, self._prev_scope):
    #   # region classification
    #   prev_cls_prob, prev_bbox_pred = self._region_classification(fc7_2, is_training,
    #                                                     initializer, initializer_bbox, postfix='_prev')

    # tracks_pred = slim.fully_connected(tf.concat([fc7, fc7_2], axis=1), num_outputs=cfg.TRAIN.BATCH_SIZE)
    # self._predictions['tracks'] = tracks_pred

    # return rois, cls_prob, bbox_pred, prev_rois, prev_cls_prob, prev_bbox_pred
    return rois, cls_prob, bbox_pred

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box

  # def _get_rpn_class_loss(self, postfix=''):
  #   rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape' + postfix], [-1, 2])
  #   rpn_label = tf.reshape(self._anchor_targets['rpn_labels' + postfix], [-1])
  #   rpn_select = tf.where(tf.not_equal(rpn_label, -1))
  #   rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
  #   rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
  #
  #   return tf.reduce_mean(
  #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
  #
  # def _get_rpn_bbox_loss(self, sigma_rpn=3.0, postfix=''):
  #   rpn_bbox_pred = self._predictions['rpn_bbox_pred' + postfix]
  #   rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets' + postfix]
  #   rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights' + postfix]
  #   rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights' + postfix]
  #   return self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
  #                                         rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
  #
  # def _get_rcnn_class_loss(self, postfix=''):
  #   cls_score = self._predictions["cls_score" + postfix]
  #   label = tf.reshape(self._proposal_targets["labels" + postfix], [-1])
  #   return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
  #
  # def _get_rcnn_bbox_loss(self, postfix=''):
  #   bbox_pred = self._predictions['bbox_pred' + postfix]
  #   bbox_targets = self._proposal_targets['bbox_targets' + postfix]
  #   bbox_inside_weights = self._proposal_targets['bbox_inside_weights' + postfix]
  #   bbox_outside_weights = self._proposal_targets['bbox_outside_weights' + postfix]
  #   return self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)


  def _get_rpn_class_loss(self):
    rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
    rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
    rpn_select = tf.where(tf.not_equal(rpn_label, -1))
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
    rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])

    return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

  def _get_rpn_bbox_loss(self, sigma_rpn=3.0):
    rpn_bbox_pred = self._predictions['rpn_bbox_pred']
    rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
    rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
    rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
    return self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

  def _get_rcnn_class_loss(self):
    cls_score = self._predictions["cls_score"]
    label = tf.reshape(self._proposal_targets["labels"], [-1])
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

  def _get_rcnn_bbox_loss(self):
    bbox_pred = self._predictions['bbox_pred']
    bbox_targets = self._proposal_targets['bbox_targets']
    bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
    bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
    return self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

  def _get_rcnn_tracks_loss(self):
    tracks = tf.tile(tf.reshape(self._proposal_targets['tracks'], [cfg.TRAIN.BATCH_SIZE, 1]), [1, cfg.TRAIN.BATCH_SIZE])
    prev_tracks = tf.transpose(tf.tile(tf.reshape(self._proposal_targets['tracks_prev'], [cfg.TRAIN.BATCH_SIZE, 1]), [1, cfg.TRAIN.BATCH_SIZE]))
    tracks_targets = tf.equal(tracks, prev_tracks)
    tracks_pred = self._predictions['tracks']

    return tf.losses.absolute_difference(tracks_targets, tracks_pred)

  def _add_losses(self):
    with tf.variable_scope('LOSS_' + self._tag) as scope:

      # RPN, class loss
      rpn_cross_entropy = self._get_rpn_class_loss()
      # prev_rpn_cross_entropy = self._get_rpn_class_loss(postfix='_prev')

      # RPN, bbox loss
      rpn_loss_box = self._get_rpn_bbox_loss()
      # prev_rpn_loss_box = self._get_rpn_bbox_loss(postfix='_prev')

      # RCNN, class loss
      cross_entropy = self._get_rcnn_class_loss()
      # prev_cross_entropy = self._get_rcnn_class_loss(postfix='_prev')

      # RCNN, bbox loss
      loss_box = self._get_rcnn_bbox_loss()
      # prev_loss_box = self._get_rcnn_bbox_loss(postfix='_prev')

      # tracks_loss = self._get_rcnn_tracks_loss()

      # self._losses['tracks'] = tracks_loss
      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box
      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box

      # self._losses['cross_entropy_prev'] = prev_cross_entropy
      # self._losses['loss_box_prev'] = prev_loss_box
      # self._losses['rpn_cross_entropy_prev'] = prev_rpn_cross_entropy
      # self._losses['rpn_loss_box_prev'] = prev_rpn_loss_box

      loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

      regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
      self._losses['total_loss'] = loss + regularization_loss

    return loss

  # def _region_proposal(self, net_conv, is_training, initializer, postfix=''):
  #   rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
  #                       scope="rpn_conv/3x3" + postfix)
  #   rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
  #                               weights_initializer=initializer,
  #                               padding='VALID', activation_fn=None, scope='rpn_cls_score' + postfix)
  #   # change it so that the score has 2 as its channel size
  #   rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape' + postfix)
  #   rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape" + postfix)
  #   rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred" + postfix)
  #   rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob" + postfix)
  #   rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
  #                               weights_initializer=initializer,
  #                               padding='VALID', activation_fn=None, scope='rpn_bbox_pred' + postfix)
  #   if is_training:
  #     rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois" + postfix)
  #     rpn_labels = self._anchor_target_layer(rpn_cls_score, name="anchor" + postfix, postfix=postfix)
  #     # Try to have a deterministic order for the computing graph, for reproducibility
  #     with tf.control_dependencies([rpn_labels]):
  #       rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois", postfix=postfix)
  #   else:
  #     if cfg.TEST.MODE == 'nms':
  #       rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois" + postfix)
  #     elif cfg.TEST.MODE == 'top':
  #       rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois" + postfix)
  #     else:
  #       raise NotImplementedError
  #
  #   self._predictions["rpn_cls_score" + postfix] = rpn_cls_score
  #   self._predictions["rpn_cls_score_reshape" + postfix] = rpn_cls_score_reshape
  #   self._predictions["rpn_cls_prob" + postfix] = rpn_cls_prob
  #   self._predictions["rpn_cls_pred" + postfix] = rpn_cls_pred
  #   self._predictions["rpn_bbox_pred" + postfix] = rpn_bbox_pred
  #   self._predictions["rois" + postfix] = rois
  #
  #   return rois, rpn

  def _region_proposal(self, net_conv, is_training, initializer):
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')
    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    if is_training:
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      rpn_labels = self._anchor_target_layer(rpn_cls_score, name="anchor")
      # Try to have a deterministic order for the computing graph, for reproducibility
      with tf.control_dependencies([rpn_labels]):
        rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois

    return rois, rpn

  # def _region_classification(self, fc7, is_training, initializer, initializer_bbox, postfix=''):
  #   cls_score = slim.fully_connected(fc7, self._num_classes,
  #                                      weights_initializer=initializer,
  #                                      trainable=is_training,
  #                                      activation_fn=None, scope='cls_score')
  #   cls_prob = self._softmax_layer(cls_score, "cls_prob")
  #   cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
  #   bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
  #                                    weights_initializer=initializer_bbox,
  #                                    trainable=is_training,
  #                                    activation_fn=None, scope='bbox_pred')
  #
  #   self._predictions["cls_score" + postfix] = cls_score
  #   self._predictions["cls_pred" + postfix] = cls_pred
  #   self._predictions["cls_prob" + postfix] = cls_prob
  #   self._predictions["bbox_pred" + postfix] = bbox_pred
  #
  #   return cls_prob, bbox_pred

  def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
    cls_score = slim.fully_connected(fc7, self._num_classes,
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
    cls_prob = self._softmax_layer(cls_score, "cls_prob")
    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
    bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_pred')

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred

    return cls_prob, bbox_pred

  def _image_to_head(self, is_training, reuse=None):
    raise NotImplementedError

  def _head_to_tail(self, pool5, is_training, reuse=None, prev=False):
    raise NotImplementedError

  def create_architecture(self, mode, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    # self._image_prev = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 6])
    # self._gt_boxes_prev = tf.placeholder(tf.float32, shape=[None, 6])
    self._tag = tag

    self._num_classes = num_classes
    self._mode = mode
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    assert tag != None

    # handle most of the regularizers here
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                    slim.conv2d_transpose,
                    slim.separable_conv2d,
                    slim.fully_connected],
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)): 
      rois, cls_prob, bbox_pred = self._build_network(training)
      # rois, cls_prob, bbox_pred, prev_rois, prev_cls_prob, prev_bbox_pred = self._build_network(training)
    layers_to_output = {'rois': rois}

    if testing:
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      self._predictions["bbox_pred"] *= stds
      self._predictions["bbox_pred"] += means

    else:
      self._add_losses()
      layers_to_output.update(self._losses)

    layers_to_output.update(self._predictions)

    return layers_to_output

  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError

  # only useful during testing mode
  def test_image(self, sess, image, im_info, prev_image=None):
    feed_dict = {self._image: image,
                 self._im_info: im_info}

    # if prev_image is not None:
    #   feed_dict.update({self._image_prev: prev_image})
    #
    # else:
    #   feed_dict.update({self._image_prev: image})

    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois

  def train_step(self, sess, blobs, train_op, blobs_prev=None):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    # if blobs_prev is not None:
    #   feed_dict.update({self._image_prev: blobs_prev['data'],
    #                     self._gt_boxes_prev: blobs_prev['gt_boxes']})
    # else:
    #   feed_dict.update({self._image_prev: blobs['data'],
    #                     self._gt_boxes_prev: blobs['gt_boxes']})
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                        self._losses['rpn_loss_box'],
                                                                        self._losses['cross_entropy'],
                                                                        self._losses['loss_box'],
                                                                        self._losses['total_loss'],
                                                                        train_op],
                                                                       feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss