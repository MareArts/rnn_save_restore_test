
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

#teach hello: hihell -> ihello
idx2char = ['h', 'i', 'e', 'l', 'o'] #class 5

x_data = [[0, 1, 0, 2, 3, 3]] #hihell
x_one_hot = [[[1, 0, 0, 0, 0],  #h 0
              [0, 1, 0, 0, 0],  #i 1
              [1, 0, 0, 0, 0],  #h 0
              [0, 0, 1, 0, 0],  #e 2
              [0, 0, 0, 1, 0],  #l 3
              [0, 0, 0, 1, 0]]] #l 3
y_data = [[1, 0, 2, 3, 3, 4]] #ihello

num_classes = 5 #class number
input_dim = 5   #one hot size
hidden_size = 5 #output from the lstm, 5 to directly predict one-hot
batch_size = 1  #one sentence
sequence_length = 6 #|ihello\ == 6
learning_rate = 0.1 #learning rate

#placeholder
X = tf.placeholder(
    tf.float32, [None, sequence_length, input_dim]) #X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length]) #Y label

#LSTM Cell
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)#, state_is_tuple=True)

def get_state_variables(batch_size, cell):
    state_variables = []
    state_c, state_h = cell.zero_state(batch_size, tf.float32)
    state_variables.append(tf.contrib.rnn.LSTMStateTuple(tf.Variable(state_c, trainable=False, name='c_state'),
                                                         tf.Variable(state_h, trainable=False, name='h_state')))
    return tf.contrib.rnn.LSTMStateTuple(tf.Variable(state_c, trainable=False), tf.Variable(state_h, trainable=False))

# For each layer, get the initial state. states will be a tuple of LSTMStateTuples.
states = get_state_variables(batch_size, cell)
output, _state = tf.nn.dynamic_rnn(cell, X, initial_state=states)

#FC layer
X_for_fc = tf.reshape(output, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)
# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
#loss
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
#train
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#prediction
prediction = tf.argmax(outputs, axis=2)

#saver
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        #training
        l, _, state= sess.run([loss, train, _state], feed_dict={X:x_one_hot, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_one_hot})
        print(i, "loss:", l, "prediction:", result, "true Y:", y_data)

        #print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str:", ''.join(result_str))
        #print(i,state)
        save_path = saver.save(sess, "./tmp/model.ckpt")


    print('check trainable variables')
    all_vars = tf.trainable_variables()
    for i in range(len(all_vars)):
        name = all_vars[i].name
        values = sess.run(name)
        print('name', name)
        print('value', values)
        print('shape',values.shape)

    #####################################################
    #value initial for checking restore
    sess.run(tf.global_variables_initializer())
    save_path = saver.restore(sess, "./tmp/model.ckpt")

    print('check trainable variables')
    all_vars = tf.trainable_variables()
    for i in range(len(all_vars)):
        name = all_vars[i].name
        values = sess.run(name)
        print('name', name)
        print('value', values)
        print('shape',values.shape)

    print('prediction result')
    result = sess.run(prediction, feed_dict={X: x_one_hot})
    print("loss:", l, "prediction:", result, "true Y:", y_data)

    # print char using dic
    result_str = [idx2char[c] for c in np.squeeze(result)]
    print("\tPrediction str:", ''.join(result_str))



