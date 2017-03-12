# encoding: utf-8
import os
import sys
# 添加模型目录
sys.path.append("./slim")
import tensorflow as tf
import numpy as np
import batchreader
from matplotlib import pyplot as plt
from nets import inception_resnet_v2
import settings

FLAGS = settings.FLAGS
slim=tf.contrib.slim

# 计算分类正确率
def compute_accuracy(v_xs, v_ys):
    global probabilities
    y_pre = sess.run(probabilities, feed_dict={xs_images: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs_images: v_xs, ys_labels: v_ys})
    return result

# tr_labels=tf.Variable(np.zeros([batch,num_classes]),dtype=tf.float32)
# ts_labels=tf.Variable(np.zeros([batch,num_classes]),dtype=tf.float32)

xs_images=tf.placeholder(dtype=tf.float32,shape=[None , FLAGS.NET_IMAGE_SIZE_H , FLAGS.NET_IMAGE_SIZE_W , FLAGS.NET_IMAGE_SIZE_C])
ys_labels=tf.placeholder(dtype=tf.float32,shape=[None , FLAGS.classes])

# inception resnet v2网络定义
logits, _ = inception_resnet_v2.inception_resnet_v2(xs_images,
                           num_classes=FLAGS.classes,
                           is_training=True)
 
probabilities = tf.nn.softmax(logits)

# 定义交叉熵loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys_labels * tf.log(probabilities),
                                              reduction_indices=[1]))       

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver=tf.train.Saver()

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(coord=coord)  
    print("Network Starts to train")
    # batch image and labels
    for step in range(FLAGS.traintimes):
        tr_img,ts_img,tr_lab,ts_lab=sess.run([batchreader.batched_train_image(),
                                              batchreader.batched_test_image(),
                                              batchreader.batched_train_label(),
                                              batchreader.batched_test_label()])
        
        # 对labels进行格式上的调整
        tr_labels=np.zeros([FLAGS.BATCH_SIZE,FLAGS.classes])
        ts_labels=np.zeros([FLAGS.test_set_size,FLAGS.classes])
        for m in range(FLAGS.BATCH_SIZE):
            tr_labels[m][tr_lab[m]]=1
        for n in range(FLAGS.test_set_size):
            ts_labels[n][ts_lab[n]]=1
        
        # 训练
        
        sess.run(train_step,feed_dict={xs_images:tr_img , ys_labels:tr_labels})
        
        # 每10步输出正确率
        if step % 10 == 0:
            print("After ",step," steps,","total accuracy is : ",compute_accuracy(
                ts_img, ts_labels))


        if step % 101 == 0:
            saver.save(sess, FLAGS.checkpoint_path,global_step=step)
        
    coord.request_stop()  
    coord.join(threads)  

#     prediction = prediction[0, 0:]
#     sorted_inds = [i[0] for i in sorted(enumerate(-prediction),
#                                         key=lambda x:x[1])]

# names = imagenet.create_readable_names_for_imagenet_labels()
# for i in range(3):
#     index = sorted_inds[i]
#     print('Probability %0.2f => [%s]' % (probabilities[index], names[index+1]))