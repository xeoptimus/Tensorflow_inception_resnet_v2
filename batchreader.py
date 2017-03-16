# encoding: utf-8
import tensorflow as tf  
import random  
import sys

# 添加模型目录
sys.path.append("./slim")  
from preprocessing import inception_preprocessing
from tensorflow.python.framework import ops  
from tensorflow.python.framework import dtypes  
import settings
FLAGS = settings.FLAGS

def encode_label(label):  
    return int(label)  

def read_label_file(file):  
    f = open(file, "r")  
    filepaths = []  
    labels = []  
    for line in f:  
        filepath, label = line.split(",") 
        filepaths.append(filepath)  
        labels.append(encode_label(label))
    return filepaths, labels  

# 读取路径与标签 
all_filepaths, all_labels = read_label_file(FLAGS.dataset_path + FLAGS.all_labels_file)  
# test_filepaths, test_labels = read_label_file(FLAGS.dataset_path + FLAGS.test_labels_file)  

# 全路径
all_filepaths = [fp for fp in all_filepaths]  
# test_filepaths = [FLAGS.dataset_path + fp for fp in test_filepaths]  

# 整合
# all_filepaths = train_filepaths + test_filepaths  
# all_labels = train_labels + test_labels  

all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)  
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)  

# 创建自定义随机分片 
partitions = [0] * len(all_filepaths)  
TEST_SET_SIZE=int(FLAGS.TEST_DATASET_RATE*len(all_filepaths))
partitions[:TEST_SET_SIZE] = [1] * TEST_SET_SIZE 
random.shuffle(partitions)  

train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)  
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)  

# 创建输入队列  
train_input_queue = tf.train.slice_input_producer(  
                                    [train_images, train_labels],  
                                    shuffle=True)  
test_input_queue = tf.train.slice_input_producer(  
                                    [test_images, test_labels],  
                                    shuffle=True)  
  
# 读图并依据网络定义要求处理图
file_content = tf.read_file(train_input_queue[0])  
train_image = tf.image.decode_jpeg(file_content, channels=FLAGS.NUM_CHANNELS)
train_image = inception_preprocessing.preprocess_image(train_image,
                                                     FLAGS.NET_IMAGE_SIZE_H,
                                                     FLAGS.NET_IMAGE_SIZE_W,
                                                     is_training=False)  
train_label = train_input_queue[1]  
  
file_content = tf.read_file(test_input_queue[0])  
test_image = tf.image.decode_jpeg(file_content, channels=FLAGS.NUM_CHANNELS) 
test_image = inception_preprocessing.preprocess_image(test_image,
                                                     FLAGS.NET_IMAGE_SIZE_H,
                                                     FLAGS.NET_IMAGE_SIZE_W,
                                                     is_training=False)  
test_label = test_input_queue[1]  
  
# 定义张量标准 
train_image.set_shape([FLAGS.NET_IMAGE_SIZE_H, FLAGS.NET_IMAGE_SIZE_W, FLAGS.NET_IMAGE_SIZE_C])  
test_image.set_shape([FLAGS.NET_IMAGE_SIZE_H, FLAGS.NET_IMAGE_SIZE_W, FLAGS.NET_IMAGE_SIZE_C])  

# batches
train_image_batch, train_label_batch = tf.train.batch(  
                                    [train_image, train_label],  
                                    batch_size=FLAGS.TRAIN_BATCH_SIZE,  
                                    num_threads=1 
                                    )  
test_image_batch, test_label_batch = tf.train.batch(  
                                    [test_image, test_label],  
                                    batch_size=FLAGS.TEST_BATCH_SIZE,  
                                    num_threads=1  
                                    )  

# 训练集图像batch
def batched_train_image():
    return train_image_batch

# 测试集图像batch
def batched_test_image():
    return test_image_batch

# 训练集标签batch
def batched_train_label():
    return train_label_batch

# 测试级标签batch
def batched_test_label():
    return test_label_batch


      