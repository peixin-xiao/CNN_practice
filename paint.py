##
# 导入第三方包
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.mlab as mlab
batch_size = 100
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
epoch_x, epoch_y = mnist.train.next_batch(batch_size)
##
x1 = epoch_x[20, :]
x2 = np.reshape(x1, [28, 28])
x3 = epoch_x[23, :]
x4 = np.reshape(x3, [28, 28])
x5 = epoch_x[1, :]
x6 = np.reshape(x5, [28, 28])
plt.subplot(1, 3, 1)
plt.imshow(x2, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(x4, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(x6, cmap='gray')