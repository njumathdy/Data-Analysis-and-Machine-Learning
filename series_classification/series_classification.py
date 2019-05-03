# encoding: utf-8

from __future__ import print_function

import tensorflow as tf 
import random 
import numpy as np 

# 用于产生样本
class GenerateSequenceData(object):
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3, max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []

        for _ in range(n_samples):
            len = random.randint(min_seq_len, max_seq_len)
            self.seqlen.append(len)

            if random.random() < .5:
                rand_start = random.randint(0, max_value - len)
                s = [[float(i) / max_value] for i in range(rand_start, rand_start + len)]
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                s = [[float(random.randint(0, max_value))/max_value] for i in range(len)]
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])
            self.batch_id = 0

    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0

        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

# 运行时的参数
learning_rate = 0.01
training_iters = 100000
batch_size = 128
display_step = 10 # 输出相关训练信息

# 网络参数
seq_max_len = 20
n_hidden = 64
n_classes = 2

training_set = GenerateSequenceData(n_samples=1000, max_seq_len=seq_max_len)
test_set = GenerateSequenceData(n_samples=500, max_seq_len=seq_max_len)

# None的位置实际为batch_size
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])

seqlen = tf.placeholder(tf.int32, [None])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def dynamicRNN(x, seqlen, weights, biases):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    batch_size = tf.shape(outputs)[0]

    # tf.gather(): 用一个一维的索引数组，将张量中对应索引的向量提取出来
    # tf.reshape(): 调整张量的维度
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    return tf.matmul(outputs, weights['out']) + biases['out']

# logits
pred = dynamicRNN(x, seqlen, weights, biases)

# 因为pred是logits，因此用tf.nn.softmax_cross_entropy_with_logits来定义损失
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# 分类准确率
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1)) # 逐个比较
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()

# 训练
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = training_set.next(batch_size)
        # 每run一次就会更新一次参数
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            # 在这个batch内计算准确度
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # 在这个batch内计算损失
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # 最终，我们在测试集上计算一次准确度
    test_data = test_set.data
    test_label = test_set.labels
    test_seqlen = test_set.seqlen
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))