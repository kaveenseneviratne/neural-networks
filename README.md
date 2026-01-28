# Fashion MNIST Neural Network (TensorFlow)

This project implements a simple **feedforward neural network** using **TensorFlow and Keras** to classify images from the **Fashion MNIST** dataset.

The model is trained to recognize 10 different categories of clothing items based on grayscale images.

---

## Dataset

**Fashion MNIST** is a dataset of 70,000 grayscale images (28x28 pixels):

- 60,000 training images
- 10,000 testing images

### Classes
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

The dataset is loaded directly from Keras:
```python
keras.datasets.fashion_mnist
```

---

## Model Architecture

The neural network consists of:

- **Flatten Layer**  
  Converts each 28x28 image into a 784-dimensional vector

- **Dense Hidden Layer**
  - 128 neurons
  - ReLU activation

- **Output Layer**
  - 10 neurons
  - Softmax activation for multi-class classification

---

## Training Details

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy
- Epochs: 25

---

## Evaluation

After training, the model is evaluated on the test dataset and prints the final test accuracy.

The script also:
- Generates predictions on test images
- Displays the first 5 test images
- Shows the actual label vs predicted label using Matplotlib

---

## How to Run

1. Install dependencies:
```bash
pip install tensorflow numpy matplotlib
```

2. Run the script:
```bash
python fashion_mnist_nn.py
```

---

## Output

- Training accuracy and loss per epoch
- Final test accuracy
- Visualization of predictions for sample images

---

## Notes

- This project is intended for learning and demonstration purposes
- No data preprocessing is required as Fashion MNIST is pre-formatted
- The model can be extended with additional layers or regularization for better performance

---

## Author
Kaveen Seneviratne
