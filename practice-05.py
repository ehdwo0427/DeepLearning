import tensorflow as tf

x_data = [1, 2, 3, 5]
y_data = [1, 2, 3, 5]

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, optimizer], feed_dict={X: [1, 2, 3, 5], Y: [1, 2, 3, 5]})
        if step & 20 == 0:
            print(step, cost_val, W_val, b_val)
    print(sess.run(hypothesis, feed_dict={X: [4]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))