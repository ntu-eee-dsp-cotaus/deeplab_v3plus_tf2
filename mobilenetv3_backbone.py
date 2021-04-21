import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from tensorflow.python.keras import backend
from tensorflow.python.keras import models
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

layers = VersionAwareLayers()

from tensorflow.keras.applications.mobilenet_v3 import *

def relu(x):
  return layers.ReLU()(x)


def hard_sigmoid(x):
  return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
  return layers.Multiply()([hard_sigmoid(x), x])

def MobileNetV3(stack_fn,
                last_point_ch,
                input_shape=None,
                alpha=1.0,
                model_type='large',
                minimalistic=False,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                classes=1000,
                pooling=None,
                dropout_rate=0.2,
                classifier_activation='softmax'):

    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    layer_utils.get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError('input_tensor: ', input_tensor,
                                 'is not type input_tensor')
        if is_input_t_tensor:
            if backend.image_data_format == 'channels_first':
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape, 'and input_tensor: ',
                                     input_tensor,
                                     'do not meet the same shape requirements')
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape, 'and input_tensor: ',
                                     input_tensor,
                                     'do not meet the same shape requirements')
        else:
            raise ValueError('input_tensor specified: ', input_tensor,
                             'is not a keras tensor')

    # If input_shape is None, infer shape from input_tensor
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError('input_tensor: ', input_tensor, 'is type: ',
                             type(input_tensor), 'which is not a valid type')

        if backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == 'channels_first':
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
                input_shape = (3, cols, rows)
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]
                input_shape = (cols, rows, 3)
    # If input_shape is None and input_tensor is None using standart shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 3)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError('Input size must be at least 32x32; got `input_shape=' +
                         str(input_shape) + '`')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25

    x = img_input
    x = layers.Rescaling(1. / 255.)(x)
    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv/BatchNorm')(x)
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)
    x = layers.Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv_1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv_1/BatchNorm')(x)
    x = activation(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='MobilenetV3' + model_type)

    return model

def MobileNetV3Large(input_shape=None,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     classes=1000,
                     pooling=None,
                     dropout_rate=0.2,
                     classifier_activation='softmax'):

    def stack_fn(x, kernel, activation, se_ratio):

        def depth(d):
            return _depth(d * alpha)

        x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
        x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
        x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
        x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
        x = _inverted_res_block(x, 3, depth(80), kernel, 2, se_ratio, activation,
                                12)
        x = _inverted_res_block(x, 6, depth(80), kernel, 1, se_ratio, activation,
                                13)
        x = _inverted_res_block(x, 6, depth(80), kernel, 1, se_ratio, activation,
                            14)
        return x

    return MobileNetV3(stack_fn, 1280, input_shape, alpha, 'large', minimalistic,
                     include_top, weights, input_tensor, classes, pooling,
                     dropout_rate, classifier_activation)

def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(
        inputs)
    if backend.image_data_format() == 'channels_first':
        x = layers.Reshape((filters, 1, 1))(x)
    else:
        x = layers.Reshape((1, 1, filters))(x)
    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv')(
            x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv_1')(
            x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
                        activation, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'expand')(
                x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand/BatchNorm')(
                x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            name=prefix + 'depthwise/pad')(
                x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=False,
        name=prefix + 'depthwise')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise/BatchNorm')(
            x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'project')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project/BatchNorm')(
            x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
        
    return x

def MobileNetV3LargeBackbone(v1ckpt, trainable=False):
    
    def tf1_tf2_name_parser(tf2name):
        newname = tf2name.split(':0')[0]

        newname = newname.replace('kernel', 'weights')
        newname = newname.replace('bias', 'biases')

        newname = f"MobilenetV3/{newname}"

        return newname
    
    LAST_LAYER = 270
    
    
    
    mn3 = MobileNetV3Large(include_top=False)
                    
    reader = tf.train.load_checkpoint(v1ckpt)
    v1layers = sorted(reader.get_variable_to_shape_map().keys())
    
    not_found = []
    not_used = [l for l in v1layers if ("MobilenetV3/" in l)]

    for i, l in enumerate(mn3.layers):
        if l.weights:
            for w in l.weights:
                tf1name = tf1_tf2_name_parser(w.name)
                
                pretrained = reader.get_tensor(tf1name)
                
                w.assign(pretrained)

                if tf1name not in v1layers:
                    not_found.append((w.name, tf1name))
                else:
                    not_used.remove(tf1name)

    not_used = [l for l in not_used if "Momentum" not in l]
    assert not not_used
    assert not not_found

    mn3.trainable = trainable
                    
    return mn3

if __name__ == "__main__":
    mn3 = MobileNetV3LargeBackbone(
        '/home/karn/deeplab_v3plus_tf2/deeplab_mnv3_large_cityscapes_trainfine/model.ckpt',
        trainable=False
    )

    print(mn3.summary())