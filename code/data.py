
import tensorgraph as tg
import tensorgraph.dataset.mnist as mnist_data
import tensorflow as tf


def cifar10(create_tfrecords=True, batch_size=32):
    tfrecords = tg.utils.MakeTFRecords()
    tfpath_train = './cifar10_train.tfrecords'
    tfpath_test = './cifar10_test.tfrecords'
    if create_tfrecords:
        X_train, y_train, X_test, y_test = tg.dataset.Cifar10()
        tfrecords.make_tfrecords_from_arrs(data_records={'X':X_train, 'y':y_train}, save_path=tfpath_train)
        tfrecords.make_tfrecords_from_arrs(data_records={'X':X_test, 'y':y_test}, save_path=tfpath_test)

    # read tfrecords
    nr_train = tfrecords.read_and_decode(tfrecords_filename_list=[tfpath_train],
                                              data_shapes={'X':[32,32,3], 'y':[10]},
                                              batch_size=batch_size)
    nr_test = tfrecords.read_and_decode(tfrecords_filename_list=[tfpath_test],
                                              data_shapes={'X':[32,32,3], 'y':[10]},
                                              batch_size=batch_size)

    n_train = sum(1 for _ in tf.python_io.tf_record_iterator(tfpath_train))
    n_test = sum(1 for _ in tf.python_io.tf_record_iterator(tfpath_test))
    return dict(nr_train), n_train, dict(nr_test), n_test


def mnist(create_tfrecords=True, batch_size=32):
    tfrecords = tg.utils.MakeTFRecords()
    tfpath_train = './mnist_train.tfrecords'
    tfpath_test = './mnist_test.tfrecords'
    if create_tfrecords:
        X_train, y_train, X_test, y_test = tg.dataset.Mnist()
        tfrecords.make_tfrecords_from_arrs(data_records={'X':X_train, 'y':y_train}, save_path=tfpath_train)
        tfrecords.make_tfrecords_from_arrs(data_records={'X':X_test, 'y':y_test}, save_path=tfpath_test)

    # read tfrecords
    nr_train = tfrecords.read_and_decode(tfrecords_filename_list=[tfpath_train],
                                              data_shapes={'X':[28,28,1], 'y':[10]},
                                              batch_size=batch_size)
    nr_test = tfrecords.read_and_decode(tfrecords_filename_list=[tfpath_test],
                                              data_shapes={'X':[28,28,1], 'y':[10]},
                                              batch_size=batch_size)

    n_train = sum(1 for _ in tf.python_io.tf_record_iterator(tfpath_train))
    n_test = sum(1 for _ in tf.python_io.tf_record_iterator(tfpath_test))
    return dict(nr_train), n_train, dict(nr_test), n_test
