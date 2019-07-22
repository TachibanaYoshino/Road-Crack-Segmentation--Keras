#coding=utf-8

from keras import regularizers
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Deconv2D,Concatenate,Input
from keras import backend as K
from keras.models import Model

K.set_image_data_format('channels_last')
bn_axis = 3




def conv_block(input_tensor, filters, kernel_size, name, strides, padding='same', dila=1):
    x = Conv2D(filters, kernel_size, strides=strides, name= name, padding=padding, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5), dilation_rate=dila)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='bn_'+ name)(x)
    x = Activation('relu')(x)

    return x

# -----

def Net(input_shape = (224,224,3)):
    inputs = Input(input_shape)
    # ---------left branch -----
    x = conv_block(inputs, 32, (3, 3), strides=1, name='L_conv1-1')
    L1 = conv_block(x, 32, (3, 3), strides=1, name='L_conv1-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L1)
    #   224 -> 112

    x  = conv_block(x, 64, (3, 3), strides=1, name='L_conv2-1')
    L2 = conv_block(x, 64, (3, 3), strides=1, name='L_conv2-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L2)
    #   112 -> 56

    x  = conv_block(x, 128, (3, 3), strides=1, name='L_conv3-1')
    L3 = conv_block(x, 128, (3, 3), strides=1, name='L_conv3-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L3)
    #   56 -> 28

    x  = conv_block(x, 256, (3, 3), strides=1, name='L_conv4-1')
    L4 = conv_block(x, 256, (3, 3), strides=1, name='L_conv4-2')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(L4)
    #   28 -> 14

    x = conv_block(x, 512, (3, 3), strides=1, name='L_conv5-1')
    x = conv_block(x, 512, (3, 3), strides=1, name='L_conv5-2')
    x = conv_block(x, 512, (3, 3), strides=1, name='L_conv5-3')
    #    14

    # ---------Right branch -----
    #   14 -> 28
    x = Deconv2D(256, kernel_size=2, strides=2, padding='same',name='R_conv1-1')(x)
    x = BatchNormalization(axis=bn_axis, name='R_conv1-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1)([x, L4]), 256, (3, 3), strides=1, name='R_conv1-2')
    x = conv_block(x, 256, (3, 3), strides=1, name='R_conv1-3')


    #   28 -> 56
    x = Deconv2D(128, kernel_size=2, strides=2, padding='same', name='R_conv2-1')(x)
    x = BatchNormalization(axis=bn_axis, name='R_conv2-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1)([x, L3]), 128, (3, 3), strides=1, name='R_conv2-2')
    x = conv_block(x, 128, (3, 3), strides=1, name='R_conv2-3')


    #   56 -> 112
    x = Deconv2D(64, kernel_size=2, strides=2, padding='same', name='R_conv3-1')(x)
    x = BatchNormalization(axis=bn_axis, name='R_conv3-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1)([x, L2]), 64, (3, 3), strides=1, name='R_conv3-2')
    x = conv_block(x, 64, (3, 3), strides=1, name='R_conv3-3')


    #   112 -> 224
    x = Deconv2D(32, kernel_size=2, strides=2, padding='same', name='R_conv4-1')(x)
    x = BatchNormalization(axis=bn_axis, name='R_conv4-1_' + 'bn')(x)
    x = conv_block(Concatenate(axis=-1)([x, L1]), 32, (3, 3), strides=1, name='R_conv4-2')
    x = conv_block(x, 32, (3, 3), strides=1, name='R_conv4-3')
    feat = conv_block(x, 32, (3, 3), strides=1, padding='same', name='feat')

    final_out = Conv2D(2, (1,1),activation='sigmoid', name='final_out')(feat)

    model = Model(inputs, final_out)
    model.summary()

    return model


