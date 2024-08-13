import numpy as np
import pickle
import pandas as pd


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def rgb_to_grayscale_luminosity(images):
    # Convert RGB images to grayscale using luminosity method
    # grayscale = 0.3 * R + 0.59 * G + 0.11 * B
    return np.dot(images, [0.3, 0.59, 0.11])


def normalize_pixel_values(images):
    # Normalize pixel values to the range [0, 1]
    return images / 255.0


def process_images(output):
    # Unpickle CIFAR-10 batch files
    data_batches = []
    for i in range(1, 6):  # CIFAR-10 has 5 training batches
        data_batches.append(unpickle(f"./cifar-10-batches-py/data_batch_{i}"))

    # Concatenate data from all batches
    x_train = np.concatenate([batch[b'data'] for batch in data_batches])  # 3072 data rgb pixels
    y_train = np.concatenate([batch[b'labels'] for batch in data_batches])  # 0 to 9 labels

    num_samples = len(x_train)
    num_classes = len(np.unique(y_train))

    # Reshape and preprocess images
    x_train_reshaped = x_train.reshape(num_samples, 3, 32, 32).transpose(0, 2, 3, 1)
    # print(x_train[0][0])
    # print(x_train[0][1024])
    # print(x_train[0][2048])
    # print((x_train_reshaped[1][31][31]))
    # print(rgb_to_grayscale_luminosity((x_train_reshaped[0][0][0])))
    # print(normalize_pixel_values(rgb_to_grayscale_luminosity((x_train_reshaped[0][0][0]))))
    x_train_grayscale = rgb_to_grayscale_luminosity(x_train_reshaped)
    # print(np.round(x_train_grayscale).astype(np.int32)[0])
    # print((x_train_grayscale[0][0][0]))
    # x_train_normalized = normalize_pixel_values(x_train_grayscale)
    x_train_normalized = np.round(x_train_grayscale).astype(np.int32)

    # Convert labels to one-hot encoding
    # y_train_encoded = pd.get_dummies(y_train.flatten(), columns=[f'class_{i}' for i in range(num_classes)])

    # Concatenate image data and labels
    data = np.concatenate((y_train.reshape(-1, 1), x_train_normalized.reshape(num_samples, -1)), axis=1)

    # Create column names
    columns = ["class"] + [f'pixel_{i}' for i in range(32 * 32)]

    # Create DataFrame
    data_df = pd.DataFrame(data, columns=columns)

    print("Starting to save data")
    # Save DataFrame to CSV
    data_df.to_csv(output, index=False)
    print("finished saving")


process_images("./cifar10_output_3.csv")
