#coding=utf-8

from keras import regularizers
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Deconv2D,Concatenate,Input,Lambda,Softmax,Add,Multiply
from keras import backend as K
from keras.models import Model

K.set_image_data_format('channels_last')
bn_axis = 3

def conv_block(input_tensor, filters, kernel_size, name, strides, padding='same', dila=1):
    x = Conv2D(filters, kernel_size, strides=strides, name= name, padding=padding, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5), dilation_rate=dila)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='bn_'+ name)(x)
    x = Activation('relu')(x)

    return x

def backend_expand_dims_1(x):
    return K.expand_dims(x, axis=1)

def backend_expand_dims_last(x):
    return K.expand_dims(x, axis=-1)

def backend_dot(x):
    return K.batch_dot(x[0], x[1])

def global_context_block(x, channels):
    bs, h, w, c = x.shape.as_list()
    input_x = x
    input_x = Reshape((h * w, c))(input_x)  # [N, H*W, C]
    input_x = Permute((2,1))(input_x)       # [N, C, H*W]
    input_x = Lambda(backend_expand_dims_1,name='a')(input_x)  # [N, 1, C, H*W]

    context_mask = Conv2D(1,(1,1), name='gc-conv0')(x)
    context_mask = Reshape((h * w, 1))(context_mask) # [N, H*W, 1]
    context_mask = Softmax(axis=1)(context_mask)  # [N, H*W, 1]
    context_mask = Permute((2,1))(context_mask)   # [N, 1, H*W]
    context_mask = Lambda(backend_expand_dims_last,name='b')(context_mask) # [N, 1, H*W, 1]

    context = Lambda(backend_dot,name='c')([input_x, context_mask])
    context = Reshape((1,1,c))(context) # [N, 1, 1, C]

    context_transform = conv_block(context, channels, 1, strides=1, name='gc-conv1')
    context_transform = Conv2D(c,(1,1), name='gc-conv2')(context_transform)
    context_transform = Activation('sigmoid')(context_transform)
    x = Multiply()([x , context_transform])

    context_transform = conv_block(context, channels, 1, strides=1, name='gc-conv3')
    context_transform = Conv2D(c,(1,1), name='gc-conv4')(context_transform)
    x = Add()([x,context_transform])

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

    # --- GC_block
    x = global_context_block(x, channels=64)

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


