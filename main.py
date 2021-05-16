#########################UNET#############################

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import PIL
from PIL import Image
import glob 
import os
import numpy as np
from numpy import asarray
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from matplotlib import pyplot
import sys
import numpy
#numpy.set_printoptions(threshold=sys.maxsize)

#import data
os.chdir(r'/content/drive/My Drive/Colab Notebooks/keras_png_slices_data')
#from tensorflow.keras.preprocessing import image_dataset_from_directory
print(os.getcwd()) 

filelist=glob.glob('keras_png_slices_seg_test/*')
segimages=np.array([np.array(Image.open(i),dtype="float32") for i in filelist])
print(segimages.shape)

filelist=glob.glob('keras_png_slices_seg_train/*')
segtrainimages=np.array([np.array(Image.open(i),dtype="float32") for i in filelist])
print(segtrainimages.shape)

filelist=glob.glob('keras_png_slices_train/*')
trainimages=np.array([np.array(Image.open(i),dtype="float32") for i in filelist])
print(np.size(filelist))
print(trainimages.shape)

filelist=glob.glob('keras_png_slices_test/*')
testimages=np.array([np.array(Image.open(i),dtype="float32") for i in filelist])
print(testimages.shape)




filelist=glob.glob('keras_png_slices_seg_validate/*')
segvalimages=np.array([np.array(Image.open(i),dtype="float32") for i in filelist])
print(np.size(filelist))

print(segvalimages.shape)

filelist=glob.glob('keras_png_slices_validate/*')
valimages=np.array([np.array(Image.open(i),dtype="float32") for i in filelist])
print(np.size(filelist))
print(valimages.shape)

segimages=(segimages)/255
testimages=(testimages)/255
trainimages=(trainimages)/255
segtrainimages=(segtrainimages)/255
segvalimages=(segvalimages)/255
valimages=(valimages)/255

#4 channels

segimages = segimages[:, :, :, np.newaxis]
testimages = testimages[:,:,:,np.newaxis]
trainimages = trainimages[:, :, :, np.newaxis]
segtrainimages = segtrainimages[:,:,:,np.newaxis]
segvalimages = segvalimages[:,:,:,np.newaxis]
valimages = valimages[:,:,:,np.newaxis]
print(segimages.shape)
print(testimages.shape)
print(trainimages.shape)
print(segtrainimages.shape)
print(segvalimages.shape)
print(valimages.shape)


#level1
#256
inputs = keras.Input(shape=(256,256,1), name="howdy")

# Apply some convolution and pooling layers
l1 = layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid",activation="relu")(inputs)
l1 = layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid", activation="relu")(l1)

l1_2 = layers.MaxPooling2D(pool_size=(2,2))(l1)
#level2
l2 = layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid", activation="relu")(l1_2)
l2 = layers.Conv2D(filters=128, kernel_size=(3,3), padding="valid", activation="relu")(l2)
#l2 = layers.Conv2D(filters=128, kernel_size=(3,3), padding="valid", activation="relu")(l2)

l2_3 = layers.MaxPooling2D(pool_size=(2,2))(l2)
#level3
l3 = layers.Conv2D(filters=128, kernel_size=(4,4), padding="valid", activation="relu")(l2_3)
l3 = layers.Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation="relu")(l3)
l3_connection = layers.Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation="relu")(l3)

l3_4 = layers.MaxPooling2D(pool_size=(2,2))(l3)
#level4
l4 = layers.Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation="relu")(l3_4)
l4 = layers.Conv2D(filters=512, kernel_size=(3,3), padding="valid", activation="relu")(l4)

l4_5 = layers.MaxPooling2D(pool_size=(2,2))(l4)
#level5

l5 = layers.Conv2D(filters=512, kernel_size=(3,3), padding="valid", activation="relu")(l4_5)
l5 = layers.Conv2D(filters=1024, kernel_size=(3,3), padding="valid", activation="relu")(l5)

#level4
l6 = layers.Conv2DTranspose(filters=512,kernel_size=(2,2), strides=(2,2),padding="valid")(l5)
#l6 = layers.UpSampling2D(size=(2,2),interpolation='nearest')(l5)


l4_connection = layers.Cropping2D(cropping=((4,4), (4,4)))(l4)
print(l6.shape)
print(l4_connection.shape)
print(l4.shape)
l6= tf.keras.layers.concatenate([l6,l4_connection])
print(l6.shape)
#l6 = layers.Conv2D(filters=1024, kernel_size=(3,3), padding="valid", activation="relu")(l6)
l6 = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(l6)
l6 = layers.Conv2D(filters=256, kernel_size=(2,2), padding="same", activation="relu")(l6)

#level3
l7 = layers.UpSampling2D(size=(2,2),interpolation='nearest')(l6)
#l7= layers.Conv2DTranspose(filters=256,kernel_size=(2,2), strides=(2,2),padding="valid")(l6)
l3_connection = layers.Cropping2D(cropping=((12,12), (12,12)))(l3)
print("connect",l7.shape,l3_connection.shape)
newl7= tf.keras.layers.concatenate([l7,l3_connection])
print("l71",newl7.shape)
l7 = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(newl7)
print("l72",l7.shape)
l7 = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(l7)
print("l73",l7.shape)

#level2
print(l7.shape)
l8= layers.Conv2DTranspose(filters=128,kernel_size=(2,2), strides=(2,2),padding="valid")(l7)
#l8= layers.UpSampling2D(size=(2,2),interpolation='nearest')(l7)
l2_connection = layers.Cropping2D(cropping=((29,29), (29,29)))(l2)
print("l8",l8.shape)
print("l2 connect",l2_connection.shape)
newl8= tf.keras.layers.concatenate([l8,l2_connection])
print("l8 concat",l8.shape)
l8 = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(newl8)
l8 = layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(l8)


#level1
#l9 = layers.UpSampling2D(size=(2,2),interpolation='nearest')(l8)
l9= layers.Conv2DTranspose(filters=64,kernel_size=(2,2), strides=(2,2),padding="valid")(l8)
l1_connection = layers.Cropping2D(cropping=((62,62), (62,62)))(l1)
print("l9",l8.shape)
print("l1 connect",l1_connection.shape)
newl9= tf.keras.layers.concatenate([l9,l1_connection])
print("l9 concat",l9.shape)

#l9= concatenate([l9,l1])
l9 = layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(newl9)
l9 = layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(l9)
l9= layers.Conv2DTranspose(filters=64,kernel_size=(2,2), strides=(2,2),padding="valid")(l9)
l9=tf.keras.layers.BatchNormalization()(l9)
l9 = layers.Conv2D(filters=4, kernel_size=(1,1), padding="same", activation="softmax")(l9)

#one more layer
#x = layers.Dropout(0.2, noise_shape=None, seed=None)(x)

#l9 = tf.keras.layers.Flatten()(l9) 
#x = layers.Conv2D(4, 3, activation="softmax", padding="same")(x)

#x = layers.Dense(4, activation="sigmoid", name="dense_2")(l9)
#outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=l9)
model.summary()
model.compile(
    optimizer='adam',
              loss='sparse_categorical_crossentropy',
    metrics=["accuracy"],
)

#print(Image.open(segtrainimages[1]))
print(segtrainimages[1,:,:,0].shape)
print(segtrainimages[1,:,:,0])

plt.imshow(z, interpolation='nearest')
plt.show()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.layers import LeakyReLU
import math
def step_decay(epoch):
	initial_lrate = 0.05
	drop = 0.2
	epochs_drop = 2.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model_5',verbose=1, save_best_only=True)
lrate=LearningRateScheduler(step_decay)
                   
print("X_train shape:", trainimages.shape)
print("y_train shape:", segtrainimages.shape)
history = model.fit(trainimages, segtrainimages, batch_size=64,epochs=30, validation_data=(valimages,segvalimages),
                    callbacks=[earlystopper, checkpointer,lrate])

history.history

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(testimages, segimages, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(testimages[:10])
print("predictions shape:", predictions.shape)
print(predictions)
print(segimages[:10])


