##Yingchen 20180314
import numpy as np
import tensorflow as tf
from model import IR_trim
from utils import read_records
import logging, os

# logging.basicConfig(format='%(module)s.%(funcName)s %(lineno)d:%(message)s', level=logging.INFO)
# logger = logging.getLogger(__name__)

IMG_SIZE_PX = 136
SLICE_COUNT = 18
batch_size = 5

filepath = '../data/cropped'
tfrecords_path_train = '../data/train_helpgs.tfrecords'
tfrecords_path_val = '../data/val_helpgs.tfrecords'
tfrecords_path_test = '../data/test_helpgs.tfrecords'

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.int64)

def read_data_all(path):
    alldata = []
    for p in os.listdir(path):
        p = os.path.join(path, p)
        if('.npy' in p and 'original' in p):
            data = np.load(p)
            alldata.append(data)
    dataarray = np.vstack(alldata)
    np.random.shuffle(dataarray)
    return dataarray

def train(x, num_epochs = 10):

    logits_train = IR_trim(x, is_training=False, reuse=False)
    logits_test = IR_trim(x, is_training=False, reuse=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=y))
    train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        best_valid_acc = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            np.random.shuffle(train_data)
            num_batches = train_data.shape[0]//batch_size
            index = 0
            for i in range(num_batches):
                batch = train_data[index:index+batch_size,:]
                index = index+batch_size
                X = [b[0] for b in batch]
                Y = [b[1] for b in batch]
                _, c = sess.run([train_op, loss], feed_dict={x: X, y: Y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of',num_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(logits_test, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Training Accuracy:',accuracy.eval({x:[i[0] for i in train_data[:200]], y:[i[1] for i in train_data[:200]]}))
            val_acc = accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]})
            if val_acc > best_valid_acc:
                best_valid_acc = val_acc
            print('Validation Accuracy:', val_acc)
        print('Done.')
        test_acc = accuracy.eval({x:[i[0] for i in test_data], y:[i[1] for i in test_data]})
        print('Best Val Accuracy:',best_valid_acc)
        print('Test Accuracy:',test_acc)

if __name__ == '__main__':
    # with tf.Graph().as_default():
    train_data = np.asarray(read_records(tfrecords_path_train))
    validation_data = np.asarray(read_records(tfrecords_path_val))
    test_data = np.asarray(read_records(tfrecords_path_test))
    print(len(train_data), len(validation_data), len(test_data))
    train(x, num_epochs=20)
