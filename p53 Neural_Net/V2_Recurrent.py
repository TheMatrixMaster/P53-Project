import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

train_data = np.loadtxt('Processed_Data/NN_2/C_Train_data.txt')
train_labels = np.loadtxt('Processed_Data/NN_2/C_Train_labels.txt')
test_data = np.loadtxt('Processed_Data/NN_2/C_Test_data.txt')
test_labels = np.loadtxt('Processed_Data/NN_2/C_Test_labels.txt')

hm_epochs = 3
n_classes = 2
batch_size = 128
chunk_size = 1
n_chunks = 32
rnn_size = 128

def next_batch(num, data, labels):

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
	prediction = recurrent_neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
   	
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    for epoch in range(hm_epochs):
	        epoch_loss = 0
	        for _ in range(int(len(train_data)/batch_size)):
	            epoch_x, epoch_y = next_batch(batch_size, train_data, train_labels)
	            epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

	            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
	            epoch_loss += c

	        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

	    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

	    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	    print('Accuracy:',accuracy.eval({x:test_data.reshape((-1, n_chunks, chunk_size)), y:test_labels}))

train_neural_network(x)