import tensorflow as tf
import numpy as np
W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
'''
init = tf.global_variables_initializer()

#网络的保存
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess,'my_net/save_net.ckpt')#自己定义的后缀
    print('Save to path:',save_path)
'''
#网络的读取restore
#重定义相同的框架shape和变量类型dtype
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')

#此处不需要initial步骤
meta_path = 'my_net/save_net.ckpt.meta'                    #图路径
model_path = './my_net/'                                     #数据路径
saver = tf.train.import_meta_graph(meta_path)              #加载图

with tf.Session() as sess:
    saver.restore(sess,model_path)    #加载数据
    print('Weights:',sess.run(W))
    print('biases:',sess.run(b))
