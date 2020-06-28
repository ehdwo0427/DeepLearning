import numpy as np
import tensorflow as tf
x_train = [1, 2, 3, 4]
y_train = [1, 2, 3, 4]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.01)

model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x_train, y_train, epochs=2000)

print(model.predict(np.array[5]))