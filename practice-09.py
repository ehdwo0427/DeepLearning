import tensorflow as tf
#queue 생성
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_default = [[0.], [0.], [0.], [0.]] #data type 선택
xy = tf.decode_csv(value, record_defaults=record_default)

train_x_batch, train_y_batch = tf.train.batch([xy[0: -1], xy[-1:]], batch_size=10) #슬라이싱해서 가져오기
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b #matrix화 선언
cost = tf.reduce_mean(tf.square(hypothesis - Y)) #loss/cost 구하는 변수

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost) #minimize한 cost/loss 찾는 optimizer변수
#학습을 위한 세션
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #초기화

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(2001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, hy_val, _ = sess.run([cost, hypothesis, optimizer], feed_dict={X: x_batch, Y: y_batch})
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
    coord.request_stop()
    coord.join(threads)