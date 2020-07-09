import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #MNIST data load

nb_classes = 10 #0~9까지의 숫자를 표현

X = tf.placeholder(tf.float32, [None, 784]) #MNIST shape 28 * 28 = 784
Y = tf.placeholder(tf.float32, [None, nb_classes]) #0~9 digits

W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, nb_classes]))
b3 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L2, W3) + b3 #matrix를 이용

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) #Adam옵티마이저 사용

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1)) #나온 값이랑 Y의 값을 비교해서 Boolean으로 반환
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) #비교 T와 F를 비교하여서 효율을 출력

training_epochs = 15 #몇번 돌것인지를 표기
batch_size = 100 #몇개로 잘라서 볼것인지를 표기

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #변수 초기화

    for epoch in range(training_epochs):
        avg_cost = 0 #cost를 확인
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch: ' '%04d' % (epoch + 1), 'cost: ', '{:.9f}'.format(avg_cost))
    print('Learning Finish')

    print('Accuracy: ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()