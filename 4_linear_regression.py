import tensorflow as tf
import numpy as np
rng = np.random

t_x = np.linspace(-2, 2, 100)
#y = 2*x^2 + 1
t_y = 2*t_x + 1 + np.random.normal(0, 0.1, 100)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(rng.randn())
b = tf.Variable(rng.randn())
pred = w*x + b
cost = tf.pow(pred - y, 2)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variable_initilizer()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(1000):
		for (x1, y1) in zip(t_x,t_y):
			sess.run(optimizer, feed_dict={x: x1, y: y1})
		if epoch % 10 == 0:
			sess.run(cost, feed_dict=

