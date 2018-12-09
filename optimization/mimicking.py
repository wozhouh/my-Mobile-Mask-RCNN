"""
Training mimic Mask R-CNN
The main Mask R-CNN model implementation.

Written by wozhouh
"""

import numpy as np
import datetime
import os
import re
import multiprocessing

# # Root directory of the project
# import sys
# ROOT_DIR = os.path.abspath("../")
# sys.path.append(ROOT_DIR)

# Import Deep learning framework
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

# Import Mask-RCNN
from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


############################################################
#  ResNet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block, prefix,
                   use_bias=True, train_bn=False):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        prefix: name prefix to distinguish teacher and student network
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = prefix + 'res' + str(stage) + block + '_branch'
    bn_name_base = prefix + 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, prefix,
               strides=(2, 2), use_bias=True, train_bn=False):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        prefix: name prefix to distinguish teacher and student network
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
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                           '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, prefix, train_bn=False):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        prefix: name prefix to distinguish teacher and student network
    """
    assert architecture in ["resnet50", "resnet101"]

    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name=prefix + 'conv1', use_bias=True)(x)
    x = BatchNorm(name=prefix + 'bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)  # output: N x 64 x 1/4 x 1/4

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', prefix=prefix, strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', prefix=prefix, train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', prefix=prefix, train_bn=train_bn)
    # output: N x 256 x 1/4 x 1/4

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', prefix=prefix, train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', prefix=prefix, train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', prefix=prefix, train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', prefix=prefix, train_bn=train_bn)
    # output: N x 512 x 1/8 x 1/8

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', prefix=prefix, train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), prefix=prefix, train_bn=train_bn)
    C4 = x
    # output: N x 1024 x 1/16 x 1/16

    # Stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', prefix=prefix, train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', prefix=prefix, train_bn=train_bn)
    C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', prefix=prefix, train_bn=train_bn)
    # output: N x 2048 x 1/32 x 1/32

    return [C1, C2, C3, C4, C5]


############################################################
#  Loss Functions
############################################################

def rpn_mimic_loss_graph(config, student_feature_maps, teacher_feature_maps, student_rpn_bbox):

    return 0.


############################################################
#  Network builder
############################################################

def build_model_body(image_shape, backbone, top_down_pyramid_size, anchor_stride, anchors_per_location,
                     is_student=False):
    # distinguish the layer name of student and teacher model by adding prefix
    if is_student:
        train_bn = True
        prefix = 's_'
    else:
        train_bn = False
        prefix = ''  # for the convenience of loading the teacher model from pre-trained weights

    # Image size must be dividable by 2 multiple times
    h, w = image_shape[:2]
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")
    # Inputs
    input_image = KL.Input(shape=[None, None, image_shape[2]], name="input_image")

    # Build the backbone of teacher/student network
    # Returns a list of the last layers of each stage, 5 in total.
    _, C2, C3, C4, C5 = resnet_graph(input_image, architecture=backbone, prefix=prefix, train_bn=train_bn)

    P5 = KL.Conv2D(top_down_pyramid_size, (1, 1), name=prefix + 'fpn_c5p5')(C5)
    P4 = KL.Add(name=prefix + "fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name=prefix + "fpn_p5upsampled")(P5),
        KL.Conv2D(top_down_pyramid_size, (1, 1), name=prefix + 'fpn_c4p4')(C4)])
    P3 = KL.Add(name=prefix + "fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name=prefix + "fpn_p4upsampled")(P4),
        KL.Conv2D(top_down_pyramid_size, (1, 1), name=prefix + 'fpn_c3p3')(C3)])
    P2 = KL.Add(name=prefix + "fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name=prefix + "fpn_p3upsampled")(P3),
        KL.Conv2D(top_down_pyramid_size, (1, 1), name=prefix + 'fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name=prefix + "fpn_p2")(P2)  # N x 256 x 256 x 256
    P3 = KL.Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name=prefix + "fpn_p3")(P3)  # N x 128 x 128 x 256
    P4 = KL.Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name=prefix + "fpn_p4")(P4)  # N x 64 x 64 x 256
    P5 = KL.Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name=prefix + "fpn_p5")(P5)  # N x 32 x 32 x 256
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name=prefix + "fpn_p6")(P5)

    # Note that P6 is used in RPN, but not in the classifier heads.
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    mrcnn_feature_maps = [P2, P3, P4, P5]

    if is_student:
        # RPN Model
        rpn = modellib.build_rpn_model(anchor_stride, len(anchors_per_location), top_down_pyramid_size)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        rpn_outputs = list(zip(*layer_outputs))
        rpn_outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(rpn_outputs, output_names)]
    else:
        rpn_outputs = None

    return KM.Model([input_image],
                    [mrcnn_feature_maps, rpn_outputs],
                    name=prefix + "model_body")


class MimicTrainingConfig(Config):
    NAME = "mimic"
    TEACHER_BACKBONE = "resnet101"
    STUDENT_BACKBONE = "resnet50"


class MimicMaskRCNN:
    def __init__(self, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(config=config)

    def build(self, config):
        """Build Mask R-CNN architecture for mimic training.
        """
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        # RPN GT
        input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        student_model_body = build_model_body(config.IMAGE_SHAPE,
                                              config.STUDENT_BACKBONE,
                                              config.TOP_DOWN_PYRAMID_SIZE,
                                              config.RPN_ANCHOR_STRIDE,
                                              len(config.RPN_ANCHOR_RATIOS),
                                              is_student=True)
        teacher_model_body = build_model_body(config.IMAGE_SHAPE,
                                              config.TEACHER_BACKBONE,
                                              config.TOP_DOWN_PYRAMID_SIZE,
                                              config.RPN_ANCHOR_STRIDE,
                                              len(config.RPN_ANCHOR_RATIOS),
                                              is_student=False)

        student_feature_maps, student_rpn_outputs = student_model_body([input_image])
        teacher_feature_maps, _ = teacher_model_body([input_image])
        student_rpn_class_logits, student_rpn_class, student_rpn_bbox = student_rpn_outputs

        anchors = self.get_anchors(config.IMAGE_SHAPE)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
        # A hack to get around Keras's bad support for constants
        anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        rpn_rois = modellib.ProposalLayer(
            proposal_count=config.POST_NMS_ROIS_TRAINING,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([student_rpn_class, student_rpn_bbox, anchors])

        # Losses
        rpn_class_loss = KL.Lambda(lambda x: modellib.rpn_class_loss_graph(*x), name="rpn_class_loss")(
            [input_rpn_match, student_rpn_class_logits])
        rpn_bbox_loss = KL.Lambda(lambda x: modellib.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
            [input_rpn_bbox, input_rpn_match, student_rpn_bbox])
        rpn_mimic_loss = KL.Lambda(lambda x: rpn_mimic_loss_graph(config, *x), name="rpn_mimic_loss")(
            [student_feature_maps, teacher_feature_maps, student_rpn_bbox])

        # Model
        inputs = [input_image, input_rpn_match, input_rpn_bbox]
        outputs = [student_feature_maps, teacher_feature_maps,
                   student_rpn_class_logits, student_rpn_class, student_rpn_bbox,
                   rpn_class_loss, rpn_bbox_loss, rpn_mimic_loss]
        model = KM.Model(inputs, outputs, name='mimic_mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)
        return model

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = modellib.compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        # Pre-defined layer regular expressions
        layer_regex = {
            "all": "s_.*",
        }

        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = modellib.data_generator(train_dataset, self.config, shuffle=True,
                                                  augmentation=augmentation,
                                                  batch_size=self.config.BATCH_SIZE,
                                                  no_augmentation_sources=no_augmentation_sources)
        val_generator = modellib.data_generator(val_dataset, self.config, shuffle=False,
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

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            modellib.log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                modellib.log("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))

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
