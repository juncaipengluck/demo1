import tensorflow as tf

a = tf.constant(10.1)
b = tf.constant(20.0)

with tf.Session() as sess:
	print("a = %f" % sess.run(a))
	print("b = %f" % sess.run(b))
	print("a + b = %f" % sess.run(a + b))
	print("a * b = %f" % sess.run(a * b))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add = tf.add(a, b)
mul = tf.multiply(a, b)
with tf.Session() as sess:
	print("a + b = %f" % sess.run(add, feed_dict={a: 1, b: 2}))
	print(" a * b = %f " % sess.run(mul, feed_dict={a: 10, b: 20}))


m1 = tf.constant([[3., 3.]])
m2 = tf.constant([[2.], [2.]])
product = tf.matmul(m1, m2)
with tf.Session() as sess:
	print(sess.run(product))
