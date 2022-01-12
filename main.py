from google.colab import drive

drive.mount('/content/drive')

"""**`Import Essental Libraries`**"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

"""**`Rescale Data`**"""

train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)

"""**`Importing Data`**"""

train_dataset = train.flow_from_directory('/content/drive/MyDrive/ZapLearnM3AI-20220112T110838Z-001/ZapLearnM3AI/Training Data',
                                          target_size = (200,200),
                                          batch_size = 3,
                                          class_mode = 'binary')

validation_dataset = validation.flow_from_directory('/content/drive/MyDrive/ZapLearnM3AI-20220112T110838Z-001/ZapLearnM3AI/Validation Data',
                                                    target_size = (200,200),
                                                    batch_size = 3,
                                                    class_mode = 'binary')

"""**`Classification`**"""

train_dataset.class_indices

train_dataset.classes

"""**`Tensorflow Keras Model`**"""

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3), activation =  'relu', input_shape = (200,200,3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2), 
                                   
                                   tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2), 
                                   
                                    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2), 

                                   tf.keras.layers.Flatten(),

                                   tf.keras.layers.Dense(512, activation = 'relu'),

                                   tf.keras.layers.Dense(1, activation = 'sigmoid'),
                                   ]
                                   )

"""**`Compiling`**"""

model.compile(loss = 'binary_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

"""**`Model Training`**"""

model_fit = model.fit(train_dataset, 
                      steps_per_epoch = 3,
                      epochs = 100,
                      validation_data = validation_dataset)

"""**`Testing Results`**"""

dir_path = '/content/drive/MyDrive/ZapLearnM3AI-20220112T110838Z-001/ZapLearnM3AI/Testing Data/Metal'

for i in os.listdir(dir_path):
  img = image.load_img(dir_path + '//' + i, target_size = (200,200))
  plt.imshow(img)
  plt.show()

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis = 0)
  images = np.vstack([x])
  val = model.predict(images)

  if val == 0:
    print("Metal")
  else:
    print("Plastic")

"""**`Random Testing`**"""

img_path = '/content/drive/MyDrive/ZapLearnM3AI-20220112T110838Z-001/ZapLearnM3AI/BisleriWaterBottle.jpg'

img = image.load_img(img_path, target_size = (200,200))
plt.imshow(img)
plt.show()

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
images = np.vstack([x])
val = model.predict(images)

if val==0:
  print("Metal")
else:
  print("Plastic")
