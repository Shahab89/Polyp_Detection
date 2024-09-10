import os

#from glob import glob
from PIL import Image
import numpy as np
from polyp_detection.params import *
from polyp_detection.interface.main import process_data

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, GlobalAveragePooling2D, Dense, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.applications import VGG19



def vgg19_model(input_size=(100, 100, 3), num_classes=8):
    # Load VGG19 with pre-trained ImageNet weights, excluding the top layers
    if os.path.isfile("vgg19_model.h5"):
    # load model
      base_model = load_model("vgg19_model.h5")
    else:
      base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_size)

    # Freeze the layers of VGG19 to retain the pre-trained weights
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the base model
    x = base_model.output
    x = Flatten()(x)

    outputs = Dense(num_classes, activation='softmax')(x)  # Final output layer for 8 classes

    # Create the model
    model = Model(inputs=base_model.input, outputs=outputs)

    return model


def combine_model ():
    model = vgg19_model()
    sgd = SGD(learning_rate=1e-3, momentum=0.9, nesterov=False)

    model.compile(
                optimizer=sgd,
                loss="categorical_crossentropy",
                metrics=["accuracy"]
                )

    return model


def train_fit_model (model, X_train,y_train):
    es = EarlyStopping(
    patience = 5,
    restore_best_weights=True
)


    try:
        history = model.fit(
            X_train, y_train,
            validation_split=.2,
            epochs=100,
            batch_size=16,
            verbose=1,
            callbacks = [es]
        )
    except Exception as e:
        print(f"Error during model fitting: {e}")

    return model, history


def prediction (image_path, model):
    image, thresholds, labels = process_data(image_path, batch_size, chunk_num)

    y_pred = model.predict(image)
    print ('y_pred:    ', y_pred)
    print (' np.argmax(y_pred, axis=1)[0]     :  ',  np.argmax(y_pred, axis=1)[0])
    print (' np.argmax(y_pred, axis=1)     :  ',  np.argmax(y_pred, axis=1))
    predicted_class_index = np.argmax(y_pred, axis=1)[0]
    categories = [
                    'dyed-lifted-polyps',
                    'dyed-resection-margins',
                    'esophagitis',
                    'normal-cecum',
                    'normal-pylorus',
                    'normal-z-line',
                    'ulcerative-colitis',
                    'polyps'
                  ]
    predicted_class_label = categories[predicted_class_index]
    return predicted_class_label
