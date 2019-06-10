import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets 

mnist = read_data_sets('Mnist',one_hot=True)    #read_data_sets读取的是压缩包
'''
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
'''

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    #tf.truncated_normal()生成服从具有指定平均值和标准偏差的正态分布值，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #x为图片的所有信息，W权重，strides滑动步长
    #same padding会在扫描过程中不够时用0填充,valid padding会舍弃
    #strides[1,x_movement,y_movement,1]
    #strides[0]和strides[3]必须为1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

#用小步长strides结合pooling的方法保留更多原始信息
#输入x为feature map
#ksize=[batch,height,width,channels]池化窗口大小,batch和channels一般设为1
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#input设为placeholder
xs = tf.placeholder(tf.float32,[None,784])  #28*28
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])  #[-1]表示自动计算此维度[1]表示灰度图像只有一个通道，输出[n_samples,28,28,1]

#conv1 layer  卷积操作将高度拉长，height和weight压缩
W_conv1 = weight_variable([5,5,1,32])    #patch 5x5,in size 1,out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)    #hiden output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)    #output size 14x14x32

#conv2 layer
W_conv2 = weight_variable([5,5,32,64])    #patch 5x5,in size 32,out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)    #hiden output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)    #output size 7x7x64

#func1 layer   [n_samples,7,7,64]-->[n_samples,7*7*64]
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#func2 layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

#the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))