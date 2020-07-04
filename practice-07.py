import tensorflow as tf
x = [1, 2, 3]
y = [1, 2 ,3]

W = tf.Variable(5.)

hypothesis = x * W

gradient = tf.reduce_mean((W * x - y) * x) * 2

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

gvs = optimizer.compute_gradients(cost, [W])
#cost와 W를 코스트앞에 있는 그래디언트를 계산해줘.
apply_gradients = optimizer.apply_gradients(gvs)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        print(step, sess.run([gradient, W, gvs]))
        sess.run(apply_gradients)