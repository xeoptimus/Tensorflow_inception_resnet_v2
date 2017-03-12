# encoding: utf-8

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Flags train settings
flags.DEFINE_integer('BATCH_SIZE', 8, 'the number of images in a batch.')
flags.DEFINE_string('train_labels_file', '/train-labels.csv', 'path to csv file for training.')
flags.DEFINE_string('test_labels_file', '/test-labels.csv', 'path to csv file for testing.')
flags.DEFINE_string('dataset_path', 'D:/Dataset/Samples/', 'path to Dataset and csv.')
flags.DEFINE_integer('test_set_size', 30, 'test size.')
tf.app.flags.DEFINE_string('checkpoint_path', 'D:/Dataset/checkpoint/model.ckpt', "Directory where to write event logs and checkpoint")
tf.app.flags.DEFINE_integer('classes', 3, "Number of classes")
tf.app.flags.DEFINE_integer('traintimes', 10000, "Number of batches to run.")

# FLags train images settings
tf.app.flags.DEFINE_integer('IMAGE_HEIGHT_ORI', 640, "original image height")
tf.app.flags.DEFINE_integer('IMAGE_WIDTH_ORI', 480, "original image weight")
tf.app.flags.DEFINE_integer('NUM_CHANNELS', 3, "original image weight")

# FLags train inputs settings
tf.app.flags.DEFINE_integer('NET_IMAGE_SIZE_H', 229, "input image height")
tf.app.flags.DEFINE_integer('NET_IMAGE_SIZE_W', 229, "input image width")
tf.app.flags.DEFINE_integer('NET_IMAGE_SIZE_C', 3, "input image channel")

