from functools import wraps
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose

def SKConv(inputs, M=2, r=16, L=32, G=32, name='sknet'):
    inputs_shape = inputs.get_shape().as_list()
    b, h, w, filters = inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]
    d = max(filters // r, L)

    x = inputs

    xs = []
    for m in range(1, M + 1):
        if G == 1:
            _x = Conv2D(filters, 3, dilation_rate=m, padding='same',
                               use_bias=False, name='{}_conv1_{}'.format(name, m))(x)
        else:
            c = filters // G
            _x = DepthwiseConv2D(3, dilation_rate=m, depth_multiplier=c, padding='same',
                                        use_bias=False, name='{}_depthconv1_{}'.format(name, m))(x)

            _x = Reshape([h, w, G, c, c])(_x)
            _x = Lambda(lambda x: K.sum(_x, axis=-1))(_x)
            _x = Reshape([h, w, filters])(_x)

        _x = BatchNormalization(name='{}_bn1_{}'.format(name, m))(_x)
        _x = Activation('relu')(_x)

        xs.append(_x)

    U = Add()(xs)
    s = Lambda(lambda x: K.mean(x, axis=[1, 2], keepdims=True))(U)

    z = Conv2D(d, 1, name='{}_conv1'.format(name))(s)
    z = BatchNormalization(name='{}_bn1'.format(name))(z)
    z = Activation('relu')(z)

    x = Conv2D(filters * M, 1, name='{}_conv2'.format(name))(z)
    x = Reshape([1, 1, filters, M])(x)
    scale = Softmax()(x)

    x = Lambda(lambda x: K.stack(x, axis=-1))(xs)

    x = Multiply()([scale, x])
    x = Lambda(lambda x: K.sum(x, axis=-1, keepdims=False))(x)

    return x

#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def resblock_body(x, num_filters, num_blocks, id=''):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)
        # y = SKConv(y, name='{}_{}'.format(id, i))
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)
        x = Add()([x,y])
    return x

#---------------------------------------------------#
#   darknet53 的主体部分
#---------------------------------------------------#
def darknet_body(x):
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1, 'a')
    x = resblock_body(x, 128, 2, 'b')
    x = resblock_body(x, 256, 8, 'c')
    feat1 = x
    x = resblock_body(x, 512, 8, 'd')
    feat2 = x
    x = resblock_body(x, 1024, 4, 'e')
    feat3 = x
    return feat1,feat2,feat3

