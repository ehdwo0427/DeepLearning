import tensorflow as tf
import matplotlib.pyplot as plt
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = W * X + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
    print(sess.run(hypothesis, feed_dict={X: [5]})) #test 1([6.1])
    print(sess.run(hypothesis, feed_dict={X: [2.5]})) #test 2([3.5])
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]})) #test 3([2.5, 4.5])