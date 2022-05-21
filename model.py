import datetime
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

#set up paths
train_table_dir = os.path.join("./data/tables/train")
test_table_dir = os.path.join("./data/tables/test")
train_chair_dir = os.path.join("./data/chairs/train")
train_table_dir = os.path.join("./data/chair/train")


# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 46 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
            './data/train/',  # This is the source directory for training images
            classes = ['chair', 'table'],
            target_size=(200, 200),  # All images will be resized to 200x200
            batch_size=36,
            # Use binary labels
            class_mode='binary')

# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
            './data/test/',  # This is the source directory for training images
            classes = ['chair', 'table'],
            target_size=(200, 200),  # All images will be resized to 200x200
            batch_size=19,
            # Use binary labels
            class_mode='binary',
            shuffle=False)


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (200,200,3)), 
                                                                        tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                                                        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

model.compile(optimizer = tf.optimizers.Adam(),
                            loss = 'binary_crossentropy',
                            metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_generator,
                          steps_per_epoch=8,  
                          epochs=15,
                          verbose=1,
                          validation_data = validation_generator,
                          validation_steps=8,
                          callbacks=[tensorboard_callback])
