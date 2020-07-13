import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777) #reproducibility 재현성

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

#input place holders
X = tf.placeholder(tf.float32, [None, 784]) #X 그래프 생성
X_img = tf.reshape(X, [-1, 28, 28, 1]) #img에 알맞게 변환
Y = tf.placeholder(tf.float32, [None, 10]) #Y 그래프 생성

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([4, 4, 1, 32], stddev=0.01)) #[width, height, rgb, filter num]
# Conv -> (?, 28, 28, 32)
# POOl -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME') #strides 1X1에 1칸씩 움지이고, 패딩은 SAME이다.
L1 = tf.nn.relu(L1) #Lelu 입력
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max pooling으로 각 칸이 14x14로 바뀌었다.

'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor(:MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
# Conv -> (?, 14, 14, 64)
# Pool -> (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2) #Relu
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max pooling 7x7
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])

'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=flaot32)
'''

# Final Fc 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2_flat, W3) + b #logits

#define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#initialize
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #train my model
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys} #각 배치에 관련된 칸 입력
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost)) #epoch 한 바퀴당  cost 수
    print('Learning Finished!')

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #효율
    print('Accuracy', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})) #효율 출력

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Lable: ", sess.run(tf.argmax(mnist.test.labels[r: r + 1], 1))) #random수 컴퓨터 input
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r: r + 1]})) #random수 컴퓨터 output
