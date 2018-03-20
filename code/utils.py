import tensorflow as tf
import numpy as np

def _float32_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_records(filename, dataset, labels):
    writer = tf.python_io.TFRecordWriter('./data/'+filename)
    for i in range(len(dataset)):
        img = dataset[i]
        label = int(labels[i])
        height = img.shape[0]
        width = img.shape[1]
        depth = img.shape[2]
        #Create a feature
        feature = {'height': _int64_feature(height),
                   'width': _int64_feature(width),
                   'depth': _int64_feature(depth),
                   'image': _bytes_feature(img.tostring()),
                   'label': _int64_feature(label)}
        #Example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        #Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

def read_records(tfrecords_path):
    dataset = []
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_path)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                                     .int64_list
                                     .value[0])

        width = int(example.features.feature['width']
                                    .int64_list
                                    .value[0])

        depth = int(example.features.feature['depth']
                                    .int64_list
                                    .value[0])

        img_string = (example.features.feature['image']
                                      .bytes_list
                                      .value[0])

        label = int(example.features.feature['label']
                                    .int64_list
                                    .value[0])

        img_1d = np.fromstring(img_string, dtype=np.int16)
        img = img_1d.reshape((height, width, depth, 2)).astype(np.float32)
        img = np.transpose(img,[2,0,1,3])
        #Reshape label to *x2 dimension
        label_2d = np.asarray([label, 1-label])
        dataset.append((img, label_2d))
    return dataset
