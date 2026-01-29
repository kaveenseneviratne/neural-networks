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

# initiating the neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)), # input layer flattened
    keras.layers.Dense(128,activation="relu"), # hidden layer
    keras.layers.Dense(10,activation="softmax") # output layer
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# training our model
model.fit(train_images,train_labels,epochs=25)

# testing the model on the test data
test_loss, test_accuracy = model.evaluate(test_images,test_labels)
print("Tested Accuracy:", test_accuracy)

# using the model to predict
prediction = model.predict(test_images) # the predict function expects an array of values
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
