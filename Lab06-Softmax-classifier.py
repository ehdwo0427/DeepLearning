import tensorflow as tf
import numpy as np
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32) #flie load
x_data = xy[:, 0:-1] #x의 범위
y_data = xy[:, [-1]] #y의 범위

nb_classes = 7 #ont_hot에서 쓰일 클래스수

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) #shape를 [Y, onehot클래수수로 바꿔주는 과정]

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b #maxrix 선언
hypothesis = tf.nn.softmax(logits) #maxrixdp hypothesis 선언

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot) #logits에 one_hot 넣어주는 과정

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1) #one_hot에서 표기해줄 숫자
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot,  1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess: #세션 구동
    sess.run(tf.global_variables_initializer()) #
    for steps in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if steps % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step : {:5}\t Loss: {:.3f}\tACC: {:.2%}".format(steps,loss,acc))
    pred = sess.run(prediction, feed_dict={X: x_data})

    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))