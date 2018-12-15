"""
Training mimic Mask R-CNN
The main Mask R-CNN model implementation.

Written by wozhouh
"""

import numpy as np
import os
import logging
import multiprocessing

# Import Deep learning framework
import tensorflow as tf
import keras
import keras.engine as KE
import keras.layers as KL
import keras.models as KM

# Import Mask-RCNN
from mrcnn import model as modellib
from mrcnn import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  ResNet Graph for student network
############################################################

# build the ResNet-50/101 based backbone
# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def s_identity_block(input_tensor, kernel_size, filters, stage, block, prefix='s_',
                     use_bias=True, train_bn=False):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        prefix: layer name prefix to distinguish teacher and student network
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = prefix + 'res' + str(stage) + block + '_branch'
    bn_name_base = prefix + 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = modellib.BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = modellib.BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = modellib.BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name=prefix + 'res' + str(stage) + block + '_out')(x)
    return x


def s_conv_block(input_tensor, kernel_size, filters, stage, block, prefix='s_',
                 strides=(2, 2), use_bias=True, train_bn=False):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        prefix: layer name prefix to distinguish teacher and student network
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = prefix + 'res' + str(stage) + block + '_branch'
    bn_name_base = prefix + 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = modellib.BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = modellib.BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1),
                  name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = modellib.BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = modellib.BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name=prefix + 'res' + str(stage) + block + '_out')(x)
    return x


def s_resnet_graph(input_image, architecture, prefix='s_', train_bn=None):
    """Build a ResNet graph.
        input_image: input to feed the ResNet graph
        architecture: Can be resnet50 or resnet101
        prefix: layer name prefix to distinguish teacher and student network
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]

    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name=prefix + 'conv1', use_bias=True)(x)
    x = modellib.BatchNorm(name=prefix + 'bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # output: N x 64 x 1/4 x 1/4

    # Stage 2
    x = s_conv_block(x, 3, [64, 64, 256], stage=2, block='a', prefix=prefix, strides=(1, 1), train_bn=train_bn)
    x = s_identity_block(x, 3, [64, 64, 256], stage=2, block='b', prefix=prefix, train_bn=train_bn)
    C2 = x = s_identity_block(x, 3, [64, 64, 256], stage=2, block='c', prefix=prefix, train_bn=train_bn)
    # output: N x 256 x 1/4 x 1/4

    # Stage 3
    x = s_conv_block(x, 3, [128, 128, 512], stage=3, block='a', prefix=prefix, train_bn=train_bn)
    x = s_identity_block(x, 3, [128, 128, 512], stage=3, block='b', prefix=prefix, train_bn=train_bn)
    x = s_identity_block(x, 3, [128, 128, 512], stage=3, block='c', prefix=prefix, train_bn=train_bn)
    C3 = x = s_identity_block(x, 3, [128, 128, 512], stage=3, block='d', prefix=prefix, train_bn=train_bn)
    # output: N x 512 x 1/8 x 1/8

    # Stage 4
    x = s_conv_block(x, 3, [256, 256, 1024], stage=4, block='a', prefix=prefix, train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = s_identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), prefix=prefix, train_bn=train_bn)
    C4 = x
    # output: N x 1024 x 1/16 x 1/16

    # Stage 5
    x = s_conv_block(x, 3, [512, 512, 2048], stage=5, block='a', prefix=prefix, train_bn=train_bn)
    x = s_identity_block(x, 3, [512, 512, 2048], stage=5, block='b', prefix=prefix, train_bn=train_bn)
    C5 = x = s_identity_block(x, 3, [512, 512, 2048], stage=5, block='c', prefix=prefix, train_bn=train_bn)
    # output: N x 2048 x 1/32 x 1/32

    return [C1, C2, C3, C4, C5]


############################################################
#  Region Proposal Network (RPN) for student network
############################################################

def s_rpn_graph(feature_map, anchors_per_location, anchor_stride, prefix='s_'):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    prefix: layer name prefix to distinguish teacher and student network

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map is not even.
    # Shared convolution base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name=prefix + 'rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name=prefix + 'rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation("softmax", name=prefix + "rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name=prefix + 'rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def s_build_rpn_model(anchor_stride, anchors_per_location, depth, prefix='s_'):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.
    prefix: layer name prefix to distinguish teacher and student network

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth], name=prefix + "input_rpn_feature_map")
    outputs = s_rpn_graph(input_feature_map, anchors_per_location, anchor_stride, prefix=prefix)
    return KM.Model([input_feature_map], outputs, name=prefix + "rpn_model")


############################################################
#  Proposal Layer for student network
############################################################

def s_apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="s_apply_box_deltas_out")
    return result


def s_clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="s_clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class s_ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(s_ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="s_top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                            self.config.IMAGES_PER_GPU,
                                            names=["s_pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: s_apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["s_refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: s_clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["s_refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def s_nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="s_rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([boxes, scores], s_nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer for student network
############################################################

class s_PyramidROIAlign(KE.Layer):
    """Implements ROI Align on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(s_PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different levels of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)  # [batch, num_boxes, 1]
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = modellib.parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = modellib.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))  # 4+log2(roi_h * roi_w)
        roi_level = tf.squeeze(roi_level, 2)  # [batch, num_boxes]

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))  # [batch, num_boxes]
            level_boxes = tf.gather_nd(boxes, ix)  # [num_boxes_level, 4]

            # Box indices for crop_and_resize (specify which image the bbox refers to)
            box_indices = tf.cast(ix[:, 0], tf.int32)  # [num_boxes_level]

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propagation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape, method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)  # [batch * num_boxes, pool_height, pool_width, channels]

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        # box_to_level: list of shape [batch, num_boxes, 1] for every pyramid level
        box_to_level = tf.concat(box_to_level, axis=0)  # [4, batch, num_boxes, 1]
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)  # [4, batch, num_boxes, 2]

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)  # [batch, num_boxes, 7, 7, 256]]
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)
        # [batch_size, num_bboxes, pool_shape(7), pool_shape(7), FPN_channels(256)]


############################################################
#  Loss Functions
############################################################

def build_transformer_layer(depth):
    input_feature_map = KL.Input(shape=[None, None, depth], name="s_transformer_input")
    transformed_feature_map = KL.Conv2D(depth, (3, 3), padding='same', strides=(1, 1), activation='linear',
                                        name='s_transformer_conv')(input_feature_map)
    return KM.Model([input_feature_map], [transformed_feature_map], name="s_transformer_layer")


def rpn_mimic_loss_graph(student_pooled, teacher_pooled):
    return tf.reduce_mean(tf.square(student_pooled - teacher_pooled)) / 2.0


############################################################
#  Data
############################################################

def s_data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                     batch_size=1, no_augmentation_sources=None):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    no_augmentation_sources: Optional. List of sources to exclude for
        augmentation. A source is string that identifies a dataset and is
        defined in the Dataset class.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # If the image source is not to be augmented pass None as augmentation
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    modellib.load_image_gt(dataset, config, image_id, augment=augment,
                                           augmentation=None,
                                           use_mini_mask=config.USE_MINI_MASK)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    modellib.load_image_gt(dataset, config, image_id, augment=augment,
                                           augmentation=augmentation,
                                           use_mini_mask=config.USE_MINI_MASK)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = modellib.build_rpn_targets(image.shape, anchors,
                                                             gt_class_ids, gt_boxes, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = modellib.mold_image(image.astype(np.float32), config)
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox]
                yield inputs, []

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  MimicMaskRCNN Class
############################################################

class MimicMaskRCNN(modellib.MaskRCNN):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training_RPN" , "training_all" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MimicMaskRCNN, self).__init__(mode, config, model_dir)

    def build(self, mode, config):
        """Build the Mask R-CNN teacher-student architecture for mimic training.
        """
        # Input
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
        # RPN GT
        input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        # Teacher Network
        # Backbone
        _, C2, C3, C4, C5 = modellib.resnet_graph(input_image, config.TEACHER_BACKBONE,
                                                  stage5=True, train_bn=config.TRAIN_BN)

        # Top-down Layers
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)  # N x 256 x 256 x 256
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)  # N x 128 x 128 x 256
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)  # N x 64 x 64 x 256
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)  # N x 32 x 32 x 256

        # Note that P6 is used in RPN, but not in the classifier heads.
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Student Network
        s_prefix = 's_'
        # Backbone
        _, S2, S3, S4, S5 = s_resnet_graph(input_image, config.STUDENT_BACKBONE,
                                           prefix=s_prefix, train_bn=None)
        # Top-down Layers
        Q5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name=s_prefix + 'fpn_s5q5')(S5)
        Q4 = KL.Add(name=s_prefix + "fpn_q4add")([
            KL.UpSampling2D(size=(2, 2), name=s_prefix + "fpn_q5upsampled")(Q5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name=s_prefix + 'fpn_s4q4')(S4)])
        Q3 = KL.Add(name=s_prefix + "fpn_q3add")([
            KL.UpSampling2D(size=(2, 2), name=s_prefix + "fpn_q4upsampled")(Q4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name=s_prefix + 'fpn_s3q3')(S3)])
        Q2 = KL.Add(name=s_prefix + "fpn_q2add")([
            KL.UpSampling2D(size=(2, 2), name=s_prefix + "fpn_q3upsampled")(Q3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name=s_prefix + 'fpn_s2q2')(S2)])
        # Attach 3x3 conv to all Q layers to get the final feature maps.
        Q2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name=s_prefix + "fpn_q2")(
            Q2)  # N x 256 x 256 x 256
        Q3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name=s_prefix + "fpn_q3")(
            Q3)  # N x 128 x 128 x 256
        Q4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name=s_prefix + "fpn_q4")(
            Q4)  # N x 64 x 64 x 256
        Q5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name=s_prefix + "fpn_q5")(
            Q5)  # N x 32 x 32 x 256
        # Q6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        Q6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name=s_prefix + "fpn_q6")(Q5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        s_rpn_feature_maps = [Q2, Q3, Q4, Q5, Q6]
        s_mrcnn_feature_maps = [Q2, Q3, Q4, Q5]

        # RPN Model
        s_rpn = s_build_rpn_model(config.RPN_ANCHOR_STRIDE,
                                  len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE, prefix=s_prefix)
        # Loop through pyramid layers
        s_layer_outputs = []  # list of lists
        for p in s_rpn_feature_maps:
            s_layer_outputs.append(s_rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        s_output_names = ["s_rpn_class_logits", "s_rpn_class", "s_rpn_bbox"]
        s_outputs = list(zip(*s_layer_outputs))
        s_outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                     for o, n in zip(s_outputs, s_output_names)]

        s_rpn_class_logits, s_rpn_class, s_rpn_bbox = s_outputs

        # Proposals
        anchors = self.get_anchors(config.IMAGE_SHAPE)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
        # A hack to get around Keras's bad support for constants
        anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else config.POST_NMS_ROIS_INFERENCE
        s_rpn_rois = s_ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="s_ROI",
            config=config)([s_rpn_class, s_rpn_bbox, anchors])

        # ROI-Align and compare the difference on the feature maps of student and teacher model
        # Add a transformer layer before comparison
        transformer = build_transformer_layer(config.TOP_DOWN_PYRAMID_SIZE)
        s_transformed_maps = [transformer([p]) for p in s_mrcnn_feature_maps]

        # rois: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
        s_pooled = s_PyramidROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE],
                                     name="s_roi_align")([s_rpn_rois, input_image_meta] + s_transformed_maps)
        pooled = modellib.PyramidROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE],
                                          name="roi_align")([s_rpn_rois, input_image_meta] + mrcnn_feature_maps)
        # output: batch x num_boxes x config.POOL_SIZE(7) x config.POOL_SIZE(7) x config.TOP_DOWN_PYRAMID_SIZE(256)

        # Losses
        rpn_class_loss = KL.Lambda(lambda x: modellib.rpn_class_loss_graph(*x), name="rpn_class_loss")(
            [input_rpn_match, s_rpn_class_logits])
        rpn_bbox_loss = KL.Lambda(lambda x: modellib.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
            [input_rpn_bbox, input_rpn_match, s_rpn_bbox])
        rpn_mimic_loss = KL.Lambda(lambda x: rpn_mimic_loss_graph(*x), name="rpn_mimic_loss")(
            [s_pooled, pooled])

        # Model
        inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox]
        outputs = [s_rpn_class_logits, s_rpn_class, s_rpn_bbox,
                   s_rpn_rois, rpn_class_loss, rpn_bbox_loss, rpn_mimic_loss]
        model = KM.Model(inputs, outputs, name='mimic_mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss", "rpn_mimic_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        # Pre-defined layer regular expressions
        layer_regex = {
            "all": "s_.*",
        }

        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = s_data_generator(train_dataset, self.config, shuffle=True,
                                           augmentation=augmentation,
                                           batch_size=self.config.BATCH_SIZE,
                                           no_augmentation_sources=no_augmentation_sources)
        val_generator = s_data_generator(val_dataset, self.config, shuffle=True,
                                         batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        modellib.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        modellib.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=self.config.BATCH_SIZE * 3,
            workers=int(multiprocessing.cpu_count() / 8),
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)
