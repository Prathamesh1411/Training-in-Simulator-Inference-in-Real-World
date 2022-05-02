# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  #Uncomment this line if you want to train on CPU
# Import Necessary Libraries
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import sparse_categorical_crossentropy
from dataloader import load_data
import numpy as np
from tensorflow.keras.metrics import MeanIoU
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Lambda
import matplotlib.pyplot as plt 


parser = argparse.ArgumentParser(description='Input Number of Classes that you want the model to segment')
parser.add_argument('classes', type=int, nargs='+',help='an integer for number of classes')
parser.add_argument('testing', type=int, nargs='+',help='Testing Mode (By Default = 0')
parser.add_argument('batch_size', type=int, nargs='+',help='Set Batch Size')
parser.add_argument('epoch', type=int, nargs='+',help='Set Number of Epochs')
args = parser.parse_args()
N = args.classes
batch_size = args.batch_size[0]
epoch = args.epoch[0]
testing = args.testing[0]

# Weighted Categorical CrossEntropy
def weighted_categorical_crossentropy(weights):
    def wcce(y_true, y_pred):
      Kweights = K.constant(weights)
      y_true = K.cast(y_true, y_pred.dtype)
      return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce

# Recall Loss
def recall_loss(weights):
    def recall_loss_fixed(y_true,y_pred):
        y_pred_onehot = tf.one_hot(tf.argmax(y_pred,axis=3),depth = N[0])
        true_positive = tf.reduce_sum((y_pred_onehot * y_true), axis=3)
        total_target = tf.reduce_sum(y_true, axis=3)
        recall = (true_positive + 1e-5) / (total_target + 1e-5)
        Kweights = K.constant(weights)
        y_true = K.cast(y_true, y_pred.dtype)
        return (1 - recall) * K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return recall_loss_fixed


# Focal Loss
def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed

# Recall Loss for Semantic Segmentation
def recall_loss_semantic_seg(weights):
  def recall_loss_inner(y_true,y_pred): 
    N = tf.shape(y_pred)[0]
    H = y_pred.shape[1]
    W = y_pred.shape[2]
    C = y_pred.shape[3]
    predict = tf.argmax(y_pred, axis=3)
    y_predict_one_hot = tf.one_hot(tf.argmax(y_pred,axis=3),depth = C)
    y_true = K.cast(y_true, dtype = 'int32')
    y_true_one_hot_ = tf.one_hot(y_true,depth = C)
    y_true_one_hot = tf.reshape(y_true_one_hot_, [N,(H*W),C])
    y_predict_one_hot = tf.reshape(y_predict_one_hot,[N,(H*W),C])
    true_positive = tf.reduce_sum((y_predict_one_hot * y_true_one_hot), axis = 1)
    total_target = tf.reduce_sum(y_true_one_hot, axis = 1)
    recall = (true_positive) / (total_target + 1e-10)
    weight = weights / np.sum(weights)
    spares = K.sparse_categorical_crossentropy(y_true, y_pred)
    loss = (1-tf.reduce_mean(recall * weight)) * spares
    return loss
  return recall_loss_inner


# Fire Convolution Module
def fire_conv(filter_series, filter_parallel_1,filter_parallel_2,input_layer, name ,series_kernel_size = (1,1), parallel1_kernel = (1,1), parallel2_kernel = (3,3), padding_='same',activation_='relu'):
  conv_series_1 = tf.keras.layers.Conv2D(filters= filter_series, kernel_size=series_kernel_size, padding=padding_, data_format="channels_last", activation= activation_)(input_layer)
  y1 = tf.keras.layers.Conv2D(filters = filter_parallel_1, kernel_size = parallel1_kernel, padding=padding_, activation = activation_)(conv_series_1)
  y2 = tf.keras.layers.Conv2D(filters = filter_parallel_2, kernel_size = parallel2_kernel, padding=padding_, activation = activation_,name=name)(conv_series_1)
  out = concatenate([y1,y2])
  return out

# Fire De-Convolution Module
def Deconv_fire_module (filter_series,filter_deconv, filter_parallel_1,filter_parallel_2,input_layer, name ,factors = [1,2],stride = (1,2), series_kernel_size = (1,1), parallel1_kernel = (1,1), parallel2_kernel = (3,3), padding_='same',activation_='relu'):
  ksize_h = 1
  ksize_w = 2
  conv1 = tf.keras.layers.Conv2D(filters= filter_series, kernel_size=series_kernel_size, padding=padding_, data_format="channels_last", activation= activation_,name=name)(input_layer)
  deconv = tf.keras.layers.Conv2DTranspose(filters = filter_series, kernel_size = (ksize_h,ksize_w),strides = stride)(conv1)
  y1 = tf.keras.layers.Conv2D(filters = filter_parallel_1, kernel_size = parallel1_kernel, padding=padding_, activation = activation_)(deconv)
  y2 = tf.keras.layers.Conv2D(filters = filter_parallel_2, kernel_size = parallel2_kernel, padding=padding_, activation = activation_)(deconv)
  out = concatenate([y1,y2])
  return out


def getClassWeights(y_train):
  # One Hot Encoding
  # y_train_one_hot = tf.keras.utils.to_categorical(y_train,num_classes = N[0])

  # Get Class Wise Weights 
  y_train_reshaped = y_train.reshape(-1,1)
  labelencoder = LabelEncoder()
  y_train_reshaped_encoded = labelencoder.fit_transform(y_train_reshaped)
  class_weights1 = class_weight.compute_class_weight(class_weight= "balanced",classes = np.unique(y_train_reshaped_encoded),y = y_train_reshaped_encoded)
  return class_weights1

def getModel():
  # Model Building Script Starts here
  input_shape = Input(shape=(64, 1024, 5))

  conv1a = tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3), strides= (1,2), padding='same', activation='relu',name='conv1a')(input_shape)

  conv1b = tf.keras.layers.Conv2D(filters= 64, kernel_size=(1,1), strides= (1,1), padding='same', activation='relu',name='conv1b')(input_shape)

  # Max Pooling1
  x = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides = (1,2),padding = 'same',name = 'maxpool1')(conv1a)

  # Fire2
  x = fire_conv(16,64,64,x,'fire2')

  # Fire3
  x = fire_conv(16,64,64,x,'fire3')

  # Saving for Skip Connection
  fire3 = x

  #Max Pooling2
  x = tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides = (1,2),padding = 'same',name = 'maxpool2')(x)

  #Fire 4
  x = fire_conv(32,128,128,x,'fire4')

  #Fire 5
  x = fire_conv(32,128,128,x,'fire5')

  # Saving for Skip Connection
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

  # Saving for Skip Connection
  firedeconv10 = x

  # Fuze
  x = tf.keras.layers.Add()([firedeconv10, fire5])

  # Fire Deconv 11
  x = Deconv_fire_module(32,32,64,64,x,'Deconv11')

  # Saving for Skip Connection
  firedeconv11 = x

  # Fuze
  x = tf.keras.layers.Add()([firedeconv11, fire3])

  # Fire Deconv 12
  x = Deconv_fire_module(16,16,32,32,x,'Deconv12')
  firedeconv12 = x

  # Fuze 
  x = tf.keras.layers.Add()([firedeconv12, conv1a])

  # Fire Deconv 13
  x = Deconv_fire_module(16,16,32,32,x,'Deconv13')
  firedeconv13 = x

  # Fuze 
  x = tf.keras.layers.Add()([firedeconv13, conv1b])

  # Final Conv Layer
  output = tf.keras.layers.Conv2D(filters= N[0], kernel_size=(1,1), strides= (1,1), padding='same', activation='softmax',name='Conv14')(x)

  # Build Model Using Functional API
  model = tf.keras.Model(inputs = input_shape,outputs = output)
  model.summary()
  return model  

def train(x_train,y_train,model):
  class_weights = getClassWeights(y_train)
  recall_ = recall_loss_semantic_seg(class_weights)
  adam = tf.keras.optimizers.Adam()
  model.compile(optimizer = adam,loss = recall_)
  checkpointer = ModelCheckpoint(filepath = "model.weights.best.hdf5",verbose = 1, save_best_only=True)
  print("TRAINING STARTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epoch,validation_split = 0.35,callbacks= [checkpointer])
  return history, model

def showLossPlot(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, 'y', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig("loss_carla_kitti_Focal_Loss.jpg")
  plt.show()

if __name__ == '__main__':
  model = getModel()
  # Load Training Data
  
  if testing == 0:
    # x_train,y_train = load_data(path='/work/pbhamare/dataset/new_dataset',normalized = True,shuffle=True)
    x_train = np.load("/home/sagar/Coursework/DL/test_carla_kitti.npy")[20:]
    y_train = np.load("/home/sagar/Coursework/DL/test_carla_kitti_labels.npy")[20:].astype(np.int32)
    history= train(x_train,y_train,model)
    # print(type(history[0]))
    showLossPlot(history[0])
  else:
    # x_test,y_test = load_data(path='/work/pbhamare/dataset/new_dataset',normalized = True,shuffle=True)
    x_test = np.load("/home/sagar/Coursework/DL/test_carla_kitti.npy")[20:]
    y_test = np.load("/home/sagar/Coursework/DL/test_carla_kitti_labels.npy")[20:].astype(np.int32)
    model.load_weights("model.weights.best.hdf5")
    #IOU
    y_pred=model.predict(x_test)
    y_pred_argmax=np.argmax(y_pred, axis=3)
    n_classes = N[0]
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(y_test, y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)
    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])

    print("IoU for class1 is: ", class1_IoU)
    print("IoU for class2 is: ", class2_IoU)
    print("IoU for class3 is: ", class3_IoU)

    # for i in range(0,x_test.shape[0]):
    #   plt.figure()
    #   plt.subplot(121)
    #   string_path_pred = "/" + str(i) + ".png"
    #   string_path_label = "/" + str(i) + ".png"
    #   plt.imshow(y_pred_argmax[i], cmap='gray')
    #   plt.subplot(122)
    #   plt.imshow(y_test[i], cmap='gray')
    #   plt.savefig(string_path_label, dpi = 200)