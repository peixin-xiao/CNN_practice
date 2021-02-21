#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#
n_classes = 10  # 10 numerical
batch_size = 100  # batch size
train_save_path = "./train_results/model"
# place holder
x = tf.placeholder('float', [None, 28 * 28])
y = tf.placeholder('float')


#
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


#
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#
def neural_network_model(data):
    # image size 28*28 -- 14*14 -- 7*7
    # using dict to  define network variables
    weights = {
        #
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name="W_conv1"),
        #
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name="W_conv2"),
        #
        'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name="W_fc"),
        #
        'out': tf.Variable(tf.random_normal([1024, n_classes]), name="W_out")
    }

    biases = {
        #
        'b_conv1': tf.Variable(tf.random_normal([32]), name="b_conv1"),
        #
        'b_conv2': tf.Variable(tf.random_normal([64]), name="b_conv2"),
        #
        'b_fc': tf.Variable(tf.random_normal([1024]), name="b_fc"),
        #
        'out': tf.Variable(tf.random_normal([n_classes]), name="b_out")
    }
    #
    data = tf.reshape(data, [-1, 28, 28, 1])

    #
    conv1 = tf.nn.relu(conv2d(data, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    #
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    #
    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    #
    output = tf.matmul(fc, weights['out']) + biases['out']

    return output



def train_neural_network(x):
    prediction = neural_network_model(x)


    #
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    #
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #
    hm_epochs = 10
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        #
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0  #
            for _ in range(int(mnist.train.num_examples / batch_size)):
                # extract data
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # feed data
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss = epoch_loss + c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)
        #
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print("saving...")
        saver.save(sess, save_path=train_save_path)


#
train_neural_network(x)

print('hello world')

