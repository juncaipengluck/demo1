import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

def inference(x):
	with tf.name_scope('layer1'):
		w1 = tf.Variable(tf.truncated_normal([28*28, 256]))
		b1 = tf.Variable(tf.truncated_normal([256]))
		y1 = tf.matmul(x, w1) + b1
	with tf.name_scope('layer2'):
		w2 = tf.Variable(tf.truncated_normal([256, 256]))
		b2 = tf.Variable(tf.truncated_normal([256]))
		y2 = tf.matmul(y1, w2) + b2
	with tf.name_scope('out'):
		w3 = tf.Variable(tf.truncated_normal([256, 10]))
		b3 = tf.Variable(tf.random_normal([10]))
		y3 = tf.matmul(y2, w3) + b3
	return y3

x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])
pred = inference(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
train_op = tf.train.AdamOptimizer(0.05).minimize(cost)
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

epochs = 50
batch_size = 120
batch_total = int(mnist.train.num_examples/batch_size)
display_step = 100

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		for i in range(batch_total):
			xs, ys = mnist.train.next_batch(batch_size)
			_, l, acc = sess.run([train_op, cost, accuracy], feed_dict={x: xs, y:ys})	
			if i % display_step == 0:
				print('epoch = %d, i = %d, loss = %f, accuracy = %f' % (epoch, i, l, acc))  

