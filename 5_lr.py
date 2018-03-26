import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def inference(x):
	w = tf.Variable(tf.truncated_normal([28*28, 10]))
	b = tf.Variable(tf.truncated_normal([10]))
	y = tf.nn.softmax(tf.matmul(x, w) + b)
	return y	


x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])
pred = inference(x)
cost = tf.reduce_mean(- tf.reduce_sum(y * tf.log(pred), reduction_indices = 1))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1)),tf.float32))
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
train_epochs = 25
batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)
display_step = 1
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(train_epochs):		
		avg_cost = 0;
		for i in range(total_batch):
			xs , ys = mnist.train.next_batch(batch_size)
			_, c = sess.run([train_op, cost], feed_dict = {x: xs, y: ys})
			avg_cost += c/total_batch
		if (epoch + 1) % display_step == 0:
			print("epoch:%d, cost=%f" %(epoch + 1, avg_cost))
	acc = sess.run(accuracy, feed_dict = {x: mnist.test.images[:3000], y:mnist.test.labels[:3000]})	
	print("accuracy = %f" % acc)  
