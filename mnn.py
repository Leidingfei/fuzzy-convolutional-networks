import tensorflow as tf
import numpy as np
import pylab as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

NUM_FEATURES = 8

learning_rate = 1e-7
beta = 1e-3
epochs = 500
batch_size = 32
n_folds=5
num_neuron = 30

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)


cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
np.random.shuffle(cal_housing)
X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
Y_data = (np.asmatrix(Y_data)).transpose()

#0.3 times X_data
m = 3* X_data.shape[0] // 10

testX, testY = X_data[:m], Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

mean = np.mean(X_data,axis=0)
stdv = np.std(X_data,axis=0)

trainX = (trainX - mean)/ stdv
testX = (testX - mean)/ stdv

trainNum = trainX.shape[0]
testNum = testX.shape[0]

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

print("N TRAIN")
print(trainNum)
print(testNum)

print(testX)
print(testY)


x = tf.placeholder(tf.float32, [None, NUM_FEATURES])

#y_ is d
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
W1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights1')
B1 = tf.Variable(tf.zeros([num_neuron]), dtype=tf.float32, name='biases1')

W2 = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(num_neuron), dtype=tf.float32), name='weights2')
B2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases2')

L1 = tf.matmul(x,W1) + B1
H1 = tf.nn.relu(L1)

y = tf.matmul(H1,W2) + B2

regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)

loss = tf.reduce_mean(tf.square(y_ - y)) + (beta*regularizer)

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
error = tf.reduce_mean(tf.square(y_ - y))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	train_err = []
	test_err = []
	idx = np.arange(trainNum)

	for i in range(epochs):
		np.random.shuffle(idx)
		x_batch = trainX[idx]
		d_batch = trainY[idx]

		for start,end in zip(range(0,trainNum,batch_size),range(batch_size,trainNum,batch_size)):
			train_op.run(feed_dict={x:x_batch[start:end], y_:d_batch[start:end]})

		errTrain = loss.eval(feed_dict={x:trainX, y_:trainY})
		train_err.append(errTrain)

		errTest = loss.eval(feed_dict={x:testX, y_:testY})
		test_err.append(errTest)

		if i % 50 == 0:
			print('iteration %d, error: train %g, test %g' % (i, train_err[i], test_err[i]))

	idx_50 = np.random.choice(testNum, size=50)
	x_50 = testX[idx_50]
	d_50 = np.array(testY[idx_50])
	y_50 = y.eval(feed_dict={x:x_50})

	# print(type(d_50))
	# print(type(y_50))

	# print(x_50.shape)
	# print(d_50.shape)
	# print(y_50.shape)

	# print(testX.shape)
	# print(testY.shape)

	# print(y_50[0])
	# print(d_50[0])
	# print(y_50[:10])
	# print(d_50[:10])
	# print(testX[0])
	# print(testY[0])

	plt.figure(1)
	plt.title("mean square error vs epochs")
	plt.plot(range(epochs), train_err, label='Validation (Train) Error')
	plt.plot(range(epochs), test_err, label='Test Error')
	plt.xlabel('epochs')
	plt.ylabel('Validation (Training) Error')
	plt.legend(loc='upper right')
	plt.savefig('Qn1fig1a.png')

	plt.figure(2)
	plt.title("mean square error vs epochs")
	plt.plot(range(epochs), train_err, label='Validation (Train) Error')
	plt.xlabel('epochs')
	plt.ylabel('Validation (Training) Error')
	plt.legend(loc='upper right')
	plt.savefig('Qn1fig1b.png')

	plt.figure(3)
	plt.title("root mean square error vs epochs")
	plt.scatter(d_50, y_50, c='r')
	plt.scatter(d_50,d_50,c='b')
	plt.title('Scatter plot of 50 predicted VS target')
	plt.xlabel('x')
	plt.ylabel('y')

	plt.show()