import tensorflow as tf
import numpy as np

def conv3d(inputs, num_features, kernel_shape, stride=1, activation_fn=tf.nn.relu, scope=None, padding='SAME'):
    """
    input : n,d,h,w,channels
    kernel_shape : kernel_shape(3D),channels,filters
    """
    channels = inputs.get_shape()[4].value
    filter_shape = kernel_shape + [channels, num_features]
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    with tf.variable_scope(scope):
        kernel = tf.get_variable(scope+'weights',filter_shape,dtype=np.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.1),
                                 regularizer = regularizer)
        biases = tf.get_variable(scope+'bias', [num_features],
                               initializer=tf.constant_initializer(0.0),dtype=np.float32)
        conv = tf.nn.conv3d(inputs, kernel, [1,stride,stride,stride,1], padding=padding)
        bias = tf.nn.bias_add(conv, biases)
        if activation_fn:
            return tf.nn.relu(bias, name=scope)
        else:
            return bias
def max_pool3d(inputs, kernel_shape, stride=1, scope=None, padding='SAME'):
    with tf.name_scope(scope) as scope:
        out = tf.nn.max_pool3d(inputs,[1]+kernel_shape+[1],[1,stride,stride,stride,1],padding=padding)
    return out

def avg_pool3d(inputs, kernel_shape, stride=1, scope=None, padding='SAME'):
    with tf.name_scope(scope) as scope:
        out = tf.nn.avg_pool3d(inputs,[1]+kernel_shape+[1],[1,stride,stride,stride,1],padding=padding)
    return out

def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = conv3d(net, 32, [1,1,1], scope='Conv3d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = conv3d(net, 32, [1,1,1], scope='Conv3d_0a_1x1')
      tower_conv1_1 = conv3d(tower_conv1_0, 32, [3,3,3], scope='Conv3d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = conv3d(net, 32, [1,1,1], scope='Conv3d_0a_1x1')
      tower_conv2_1 = conv3d(tower_conv2_0, 48, [3,3,3], scope='Conv3d_0b_3x3')
      tower_conv2_2 = conv3d(tower_conv2_1, 64, [3,3,3], scope='Conv3d_0c_3x3')
    mixed = tf.concat(axis=4, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = conv3d(mixed, net.get_shape()[4], [1,1,1],
                     activation_fn=None, scope='Conv2d_1x1')
    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net

def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = conv3d(net, 192, [1, 1, 1], scope='Conv3d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = conv3d(net, 128, [1, 1, 1], scope='Conv3d_0a_1x1')
      tower_conv1_1 = conv3d(tower_conv1_0, 160, [1, 1, 7], scope='Conv3d_0b_1x7')
      tower_conv1_2 = conv3d(tower_conv1_1, 192, [1, 7, 1], scope='Conv3d_0c_7x1')
    mixed = tf.concat(axis=4, values=[tower_conv, tower_conv1_2])
    up = conv3d(mixed, net.get_shape()[4], [1, 1, 1],
                     activation_fn=None, scope='Conv3d_1x1')

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net

# class Classifier():
#     def __init__(self,
#                 nclass=2,
#                 dropout_keep_prob=0.8,
#                 reuse=None,
#                 scope='InceptionResNet_trim',
#                 activation_fn=tf.nn.relu):
#         self.nclass = nclass
#         self.activation_fn = activation_fn
#         self.dropout_keep_prob = dropout_keep_prob
#         self.scope = scope
#         self.reuse = reuse

def IR_trim(inputs, nclass=2,
            dropout_keep_prob=0.5,
            reuse=None,
            scope='InceptionResNet_trim',
            activation_fn=tf.nn.relu,
            is_training=True,
            padding = 'SAME'):
    with tf.variable_scope(scope, reuse=reuse):
        #Stem
        IMG_SIZE_PX = 133
        SLICE_COUNT = 18
        inputs = tf.reshape(inputs, shape=[-1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX, 1])
        net = conv3d(inputs, 32, [3,3,3], stride=2, scope='Conv3d_1a_3x3')
        net = conv3d(net, 32, [3,3,3], stride=1, scope='Conv3d_2a_3x3')
        net = conv3d(net, 64, [3,3,3], stride=1, scope='Conv3d_2b_3x3')
        net = max_pool3d(net, [3,3,3], stride=2, scope = 'MaxPool_3a_3x3')
        net = conv3d(net, 80, [1,1,1], stride=1, scope='Conv3d_3b_1x1')
        net = conv3d(net, 192, [3,3,3], stride=1, scope='Conv3d_4a_3x3')
        net = conv3d(net, 256, [3,3,3], stride=2, scope='Conv3d_5a_3x3')
        #Block35
        net = block35(net, activation_fn=activation_fn)
        #Reduce
        with tf.variable_scope('Mixed_6a'):
            with tf.variable_scope('Branch_0'):
              tower_conv = conv3d(net, 384, [3, 3, 3], stride=2,
                                       padding=padding,
                                       scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
              tower_conv1_0 = conv3d(net, 256, [1, 1, 1], scope='Conv2d_0a_1x1')
              tower_conv1_1 = conv3d(tower_conv1_0, 256, [3, 3, 3],
                                          scope='Conv2d_0b_3x3')
              tower_conv1_2 = conv3d(tower_conv1_1, 384, [3, 3, 3],
                                          stride=2,
                                          padding=padding,
                                          scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
              tower_pool = max_pool3d(net, [3,3,3], stride=2,
                                           padding=padding,
                                           scope='MaxPool_1a_3x3')
            net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 4)
        #Block17
        net = block17(net, activation_fn=None)
        net = conv3d(net, 1536, [1, 1, 1], scope='Conv2d_7b_1x1')
        #AvgPooling + Dropout + Softmax
        with tf.variable_scope('Logits'):
            net = avg_pool3d(net, [2,2,2], stride=1, padding='SAME',scope='AvgPool_1a_8x8')
            net = tf.contrib.layers.flatten(net, scope='Flatten')
            net = tf.layers.dropout(net, dropout_keep_prob, training=is_training)
            logits = tf.contrib.layers.fully_connected(net, nclass, activation_fn=None,
                                  scope='Logits')
        return logits
