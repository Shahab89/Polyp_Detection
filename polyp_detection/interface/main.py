import numpy as np
import os
from polyp_detection.params import *
from polyp_detection.ml_logic.model import *
from polyp_detection.interface.preprocessing import *
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_images_in_chunks(data_directory, batch_size, image_size=image_size):
    classes = os.listdir(data_directory)
    if '.ipynb_checkpoints' in classes:
        classes.remove('.ipynb_checkpoints')
    file_paths = []
    labels = []
    for class_index, class_name in enumerate(classes):
        class_dir = os.path.join(data_directory, class_name)
        for file_name in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, file_name))
            labels.append(class_index)

    print(classes)
    print(np.unique(labels))
    data = list(zip(file_paths, labels))
    np.random.shuffle(data)
    file_paths, labels = zip(*data)

    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_images = [preprocess_image(file, image_size) for file in batch_files]
        # Separate the images and thresholds
        batch_images, batch_thresholds = zip(*batch_images)
        batch_images = np.array(batch_images)
        batch_thresholds = np.array(batch_thresholds)
        batch_labels = np.array(batch_labels)
        # Yield the batch of images, thresholds, and labels
        yield batch_images, batch_thresholds, batch_labels


def process_data(data_directory, batch_size, chunk_num):
    # Initialize lists to store all chunks
    all_images = []
    all_thresholds = []
    all_labels = []

    # Iterate through each chunk of images, thresholds, and labels
    iter = 1
    for images, thresholds, labels in load_images_in_chunks(data_directory, batch_size):
        print(iter, ')   :  ', images.shape, thresholds.shape, labels.shape)
        all_images.append(images)
        all_thresholds.append(thresholds)
        all_labels.append(labels)

        if iter == chunk_num:
            break
        else:
            iter += 1

    # After the loop, concatenate all chunks along the first axis
    images = np.concatenate(all_images, axis=0)
    thresholds = np.concatenate(all_thresholds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    #  combine arrays for images, thresholds, and labels
    print('total data: ', images.shape, thresholds.shape, labels.shape)

    return images, thresholds, labels


if __name__ == '__main__':
    try:
        if run_model == 'from_dir':
            trained_model = load_model("models/vgg19_model_20240905_140342.h5")
        else:
            model = combine_model()

            images, thresholds, labels = process_data(data_directory, batch_size, chunk_num)
            X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
            y_train = to_categorical(y_train, num_classes=8)
            y_test = to_categorical(y_test, num_classes=8)

            print(f'X_train type: {type(X_train)}, shape: {X_train.shape}')
            print(f'y_train type: {type(y_train)}, shape: {y_train.shape}')
            print(f'X_test type: {type(X_test)}, shape: {X_test.shape}')
            print(f'y_test type: {type(y_test)}, shape: {y_test.shape}')

            trained_model, history = train_fit_model(model, X_train, y_train)

        prediction = prediction(image_path, trained_model)

        print ('prediction:    ', prediction)
    except Exception as e:
        import sys
        import traceback
        import ipdb

        print(f"An error occurred: {e}")
        traceback.print_exc()
        ipdb.post_mortem()
