import tensorflow as tf
import numpy as np

#Read_Data
train_data = np.loadtxt('Processed_Data/NN_1/Train_data.txt')
train_labels = np.loadtxt('Processed_Data/NN_1/Train_labels.txt')
test_data = np.loadtxt('Processed_Data/NN_1/Test_data.txt')
test_labels = np.loadtxt('Processed_Data/NN_1/Test_labels.txt')

n_nodes_hl1 = 15
n_nodes_hl2 = 15
n_nodes_hl3 = 15

n_classes = 2
batch_size = 100

def next_batch(num, data, labels):

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


x = tf.placeholder('float', [None, 12])
y = tf.placeholder('float')

def neural_network_model(data):

	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([12, n_nodes_hl1])),\
					'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),\
					'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),\
					'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),\
					'biases': tf.Variable(tf.random_normal([n_classes]))}

	#(input_data * weights) + biases

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	
	return output

def train_neural_network(x):
	#feedforward
	prediction = neural_network_model(x)

	#cost function with cross entropy
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))

	#reducing the cost using AdamOptimizer / learning constant = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#Feedforward + Backpropagation
	hm_epochs = 10


	#Running the Session

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(len(train_data)/batch_size)):
				epoch_x, epoch_y = next_batch(batch_size, train_data, train_labels)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)	


		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))	
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x : test_data, y : test_labels}))


train_neural_network(x)		