import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.utils import to_categorical
import cv2
import numpy as np

def load_mnist_png(data_path):
    images = []
    labels = []

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        for image_file in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, image_file), cv2.IMREAD_GRAYSCALE)
            img = img.astype('float32') / 255.0 # Normalizacja danych do wartości zmienno przecinkowych od 0 do 1
            images.append(img)

            label = int(folder)
            labels.append(label)

    return np.array(images), np.array(labels)

input_layer = Flatten(input_shape=(28, 28))

hidden_layer = Dense(units=128, activation='relu')

output_layer = Dense(units=10, activation='softmax')

model = Sequential([input_layer,
                    hidden_layer,
                    output_layer])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

x_train, y_train = load_mnist_png('DATA\\mnistpng\\mnist_png\\training')
x_test, y_test = load_mnist_png('DATA\\mnistpng\\mnist_png\\testing')

# Przekonwertowanie etykiety na kodowanie one-hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model.fit(x_train, y_train, epochs=10, batch_size=128)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])