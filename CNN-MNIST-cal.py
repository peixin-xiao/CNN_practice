#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# one-hot coding
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# variables setting
n_classes = 10  #
batch_size = 41  # batch size
train_save_path="./train_results/model"
# place holder
x = tf.placeholder('float', [None, 28 * 28])
y = tf.placeholder('float')
# sim = tf.placeholder('float')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define network
def neural_network_model(data):
    # img size 28*28 -- 14*14 -- 7*7

    data = tf.reshape(data, [-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
    conv1 = maxpool2d(conv1)

    #
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2) + b_conv2)
    conv2 = maxpool2d(conv2)

    #
    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, W_fc) + b_fc)

    #
    output = tf.matmul(fc, W_out) + b_out

    return output


# cosine similarity
def cosine_similarity(im1, im2, norm=False):

    res = np.array([[im1[i] * im2[i], im1[i] * im1[i], im2[i] * im2[i]] for i in range(len(im1))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos


with tf.Session() as sess:
    #
    # sess.run(tf.global_variables_initializer())
    saver1 = tf.train.import_meta_graph('./train_results/model.meta')
    saver1.restore(sess, save_path=train_save_path)
    graph = tf.get_default_graph()
    W_conv1 = graph.get_tensor_by_name("W_conv1:0")
    W_conv2 = graph.get_tensor_by_name("W_conv2:0")
    W_fc = graph.get_tensor_by_name("W_fc:0")
    W_out = graph.get_tensor_by_name("W_out:0")
    b_conv1 = graph.get_tensor_by_name("b_conv1:0")
    b_conv2 = graph.get_tensor_by_name("b_conv2:0")
    b_fc = graph.get_tensor_by_name("b_fc:0")
    b_out = graph.get_tensor_by_name("b_out:0")
    epoch_x, epoch_y = mnist.train.next_batch(batch_size)
    #
    # test validity of image recognition
    # y_ = tf.argmax(neural_network_model(x), 1)
    # y_label = y
    # y_, y_label = sess.run([y_, y_label], feed_dict={x: epoch_x, y: epoch_y})
    # print(y_[1], y_label[1])
    # print(y_[2], y_label[2])
    #
    # calculate the vector of each image

    v = neural_network_model(x)
    y_label = y
    v, y_label = sess.run([v, y_label], feed_dict={x: epoch_x, y: epoch_y})
    sim = np.zeros(shape=[40], dtype=float)
    for j in range(40):
        sim[j] = cosine_similarity(v[0], v[j])
    print(sim, '\n', y_label)
    # split1, split2 = tf.split(epoch_x, [20, 80], axis=0)


print('hello world')
