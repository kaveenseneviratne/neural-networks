import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# importing the fashion mnist dataset from keras
data = keras.datasets.fashion_mnist

# splitting data into training and testing splits
(train_images, train_labels),(test_images,test_labels) = data.load_data()

# defining what the labels in the dataset are
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# scaling the data
train_images = train_images/255
test_images = test_images/255

# checking one specific example of the data
plt.imshow(train_images[7], cmap = plt.cm.binary)
plt.show()
