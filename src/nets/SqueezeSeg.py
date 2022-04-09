import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy
# from dataloader import load_train_carla
import numpy as np
from sklearn.model_selection import train_test_split


def fire_conv(filter_series, filter_parallel_1,filter_parallel_2,input_layer, name ,series_kernel_size = (1,1), parallel1_kernel = (1,1), parallel2_kernel = (3,3), padding_='same',activation_='relu'):
  conv_series_1 = tf.keras.layers.Conv2D(filters= filter_series, kernel_size=series_kernel_size, padding=padding_, data_format="channels_last", activation= activation_)(input_layer)
  y1 = tf.keras.layers.Conv2D(filters = filter_parallel_1, kernel_size = parallel1_kernel, padding=padding_, activation = activation_)(conv_series_1)
  y2 = tf.keras.layers.Conv2D(filters = filter_parallel_2, kernel_size = parallel2_kernel, padding=padding_, activation = activation_,name=name)(conv_series_1)
  out = concatenate([y1,y2])
  # Fire = tf.keras.models.Model(input_layer,out)
  return out


def Deconv_fire_module (filter_series,filter_deconv, filter_parallel_1,filter_parallel_2,input_layer, name ,factors = [1,2],stride = (1,2), series_kernel_size = (1,1), parallel1_kernel = (1,1), parallel2_kernel = (3,3), padding_='same',activation_='relu'):
  # ksize_h = int(factors[0] * 2 - factors[0] % 2)
  # ksize_w = int(factors[1] * 2 - factors[1] % 2)
  # print("H = {}".format(ksize_h))
  # print("W = {}".format(ksize_w))
  ksize_h = 1
  ksize_w = 2
  conv1 = tf.keras.layers.Conv2D(filters= filter_series, kernel_size=series_kernel_size, padding=padding_, data_format="channels_last", activation= activation_,name=name)(input_layer)
  deconv = tf.keras.layers.Conv2DTranspose(filters = filter_series, kernel_size = (ksize_h,ksize_w),strides = stride)(conv1)
  y1 = tf.keras.layers.Conv2D(filters = filter_parallel_1, kernel_size = parallel1_kernel, padding=padding_, activation = activation_)(deconv)
  y2 = tf.keras.layers.Conv2D(filters = filter_parallel_2, kernel_size = parallel2_kernel, padding=padding_, activation = activation_)(deconv)
  out = concatenate([y1,y2])
  # FireDeconv = tf.keras.models.Model(input_layer,out)
  return out


model = tf.keras.Sequential()
input_shape = Input(shape=(64, 1024, 5))

conv1a = tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3), strides= (1,2), padding='same', activation='relu',name='conv1a')(input_shape)

conv1b = tf.keras.layers.Conv2D(filters= 64, kernel_size=(1,1), strides= (1,1), padding='same', activation='relu',name='conv1b')(input_shape)

# Max Pooling1
x = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides = (1,2),padding = 'same',name = 'maxpool1')(conv1a)
# conv
# Fire2
x = fire_conv(16,64,64,x,'fire2')

# Fire3
x = fire_conv(16,64,64,x,'fire3')
fire3 = x
#Max Pooling2
x = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides = (1,2),padding = 'same',name = 'maxpool2')(x)

#Fire 4
x = fire_conv(32,128,128,x,'fire4')

#Fire 5
x = fire_conv(32,128,128,x,'fire5')
fire5 = x
# Max Pooling
x = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides = (1,2),padding = 'same', name = 'maxpool3')(x)

# Fire 6
x = fire_conv(48,192,192,x,'fire6')

#Fire7
x = fire_conv(48,192,192,x,'fire7')

# Fire 8

x = fire_conv(64,256,256,x,'fire8')

# Fire 9
x = fire_conv(64,256,256,x,'fire9')

#Fire Deconv 10
x = Deconv_fire_module(64,64,128,128,x,'Deconv10')
firedeconv10 = x
# Fuze
x = tf.add(firedeconv10,fire5)

# Fire Deconv 11
x = Deconv_fire_module(32,32,64,64,x,'Deconv11')
firedeconv11 = x
# Fuze
x = tf.add(firedeconv11,fire3)

# Fire Deconv 12
x = Deconv_fire_module(16,16,32,32,x,'Deconv12')
firedeconv12 = x
# Fuze
x = tf.add(firedeconv12, conv1a)

# Fire Deconv 13
x = Deconv_fire_module(16,16,32,32,x,'Deconv13')
firedeconv13 = x
# Fuze
x = tf.add(firedeconv13,conv1b)

x = tf.keras.layers.Dropout(0.5)(x)
# Conv 14
output = tf.keras.layers.Conv2D(filters= 3, kernel_size=(1,1), strides= (1,1), padding='same', activation='softmax',name='Conv14')(x)

model = Model(inputs = input_shape,outputs = output)
# model.summary()

model.compile(optimizer = "adam",loss = sparse_categorical_crossentropy, metrics = tf.keras.metrics.MeanIoU(num_classes = 4))

checkpointer = ModelCheckpoint(filepath = "model.weights.best.hdf5",verbose = 1, save_best_only=True)

# model.fit(x_train,y_train, batch_size = 50, epochs = 50, validation_data = (x_valid,y_valid), callbacks = [checkpointer])

model.compile(optimizer = "adam",loss = sparse_categorical_crossentropy, metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

# checkpointer = ModelCheckpoint(filepath = "model.weights.best.hdf5",verbose = 1, save_best_only=True)

# model.fit(x_train,y_train, batch_size = 2, epochs = 10, validation_data = (x_val,y_val), callbacks = [checkpointer])

model.load_weights("model.weights.best.hdf5")
# score = model.evaluate(x_test,y_test, verbose=0)

# print(score)
# output = model.predict(x_test.reshape(64,1024,5))
# np.savetxt('output.txt',output)