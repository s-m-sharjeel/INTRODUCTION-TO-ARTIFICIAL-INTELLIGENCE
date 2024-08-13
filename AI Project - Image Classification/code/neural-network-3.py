import random
import numpy as np
import pandas as pd
import tkinter as tk
import pickle
from tkinter import filedialog
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

# Importing numpy to work efficiently with arrays

# Our neural network
# it has one input layer, one hidden layers, and one output layer
# the input layer has 32x32 = 1024 nodes, each representing a pixel grayscal value betweeon 0 and 1 in 32x32 images
# the first hidden layer will have 16 nodes
# the second hidden layer will have 16 nodes
# the output layer will have 10 nodes , each node for the class category

# reading the csv file provided in the drive link:
# https://drive.google.com/drive/folders/1P5FdcN6vXZk1T-4uw8KcXidOyGaIPHky?usp=drive_link
data = pd.read_csv("./cifar10_output_3.csv")

# showing the file output
# print(data.head())

data = np.array(data)  # converting the panda read file to numpy array for faster processing
m, n = data.shape  # m = rows, n = cols, for later use
np.random.shuffle(data)  # shuffling for spliting for test/train

data_dev = data[0:1000].T  # testing data, first 1000 samples
Y_dev = data_dev[0]
X_dev = data_dev[1:n]  # all data from second row to end row, change for cifar
X_dev = X_dev / 255.  # for normalziign brigtness between 0 and 1

data_train = data[10000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape  # no idea why


def init_params():
    # First hidden layer, 512 nodes
    W1 = np.random.rand(16, 1024) - 0.5  # create a 1024 array of random values, our initial weights
    b1 = np.random.rand(16, 1) - 0.5  # our bias for inputs

    # Second hidden layer, 256 nodes
    W2 = np.random.rand(16, 16) - 0.5  # wieght for 2nd row
    b2 = np.random.rand(16, 1) - 0.5  # bias for 2nd layer

    # Output Layer
    W3 = np.random.rand(10, 16) - 0.5  # 10 neurons in output layer, 256 neurons in second hidden layer
    b3 = np.random.rand(10, 1) - 0.5  # Bias for each neuron in the output layer

    return W1, b1, W2, b2, W3, b3


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(z):
    return np.exp(z) / sum(np.exp(z))


def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def derive_ReLU(Z):
    return Z > 0


def back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)

    dZ3 = A3 - one_hot_Y  # Error
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * derive_ReLU(Z2)

    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    # Compute derivative of the error with respect to the activations of the first hidden layer
    dZ1 = W2.T.dot(dZ2) * derive_ReLU(Z1)

    # Compute gradients of the weights and biases of the first layer
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    # dZ2 = A2 - one_hot_Y
    # dW2 = 1 / m * dZ2.dot(A1.T)
    # db2 = 1 / m * np.sum(dZ2)
    # dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    # dW1 = 1 / m * dZ1.dot(X.T)
    # db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 500 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2, W3, b3


def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions


def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((32, 32)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    return classes[int(prediction)]


def train_network():
    W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.4, 5000)
    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2, W3, b3)
    accuracy = get_accuracy(dev_predictions, Y_dev)
    return (accuracy, W1, b1, W2, b2, W3, b3)


def test_prediction2(data, W1, b1, W2, b2, W3, b3):
    current_image = data
    prediction = make_predictions(data, W1, b1, W2, b2, W3, b3)
    return prediction


# Function to save data
def save_arrays(tuuple):
    with open("./tuple.pkl", "wb") as f:
        pickle.dump(tuuple, f)


# Function to load data
def load_arrays():
    with open("./tuple.pkl", "rb") as f:
        tuuple = pickle.load(f)
    return tuuple


try:
    with open("tuple.pkl", "rb") as f:
        tuuple = load_arrays()
        print("Data loaded successfully.")
except FileNotFoundError:
    print("Data file not found. Training new model.")
    tuuple = train_network()
    save_arrays(tuuple)
    print("Neural Network generated and saved.")

accuracy, W1, b1, W2, b2, W3, b3 = tuuple

## UI CODE BELOW


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Function to process the image
def process_image():
    if image_label.image:
        # Access the selected image
        selected_image = image_label.image
        a = test_prediction2(preprocess_image(selected_image), W1, b1, W2, b2, W3, b3)
        output_label.config(text=f'{classes[a]}')
    else:
        output_label.config(text="Please choose an image first!")
    # Placeholder function, you can replace it with your image processing logic


# Function to open a file dialog and select an image
def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((32, 32))  # Resize the image
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img  # Keep reference to prevent garbage collection


def preprocess_image(image):
    # Open the image
    img = image

    # Resize the image to 32x32
    # img = img.resize((32, 32))

    # Convert the image to grayscale
    img = img.convert("L")

    # Convert the image to a 1D array
    img_array = list(img.getdata())

    return img_array

# Function to open a dialog box for entering a number
def open_number_dialog():
    number = random.randint(0, 10000)
    # Placeholder: Process the entered number here
    output_label.config(text=f"Image classified as: {test_prediction(number, W1, b1, W2, b2, W3, b3)}")



# Create the main window
root = tk.Tk()
root.title("Image Classification")

# Create labels
image_label = tk.Label(root)
image_label.pack(pady=10)

output_label = tk.Label(root, text="")
output_label.pack(pady=10)

# Create buttons
choose_button = tk.Button(root, text="Choose Image", command=choose_image)
choose_button.pack()

process_button = tk.Button(root, text="Process Image", command=process_image)
process_button.pack(pady=10)

# Create a button to open the number dialog
number_button = tk.Button(root, text="Select Random Testing Image", command=open_number_dialog)
number_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
