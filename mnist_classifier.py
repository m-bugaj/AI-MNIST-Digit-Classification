import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense

input_layer = Flatten(input_shape=(28, 28))

hidden_layer = Dense(units=128, activation='relu')

output_layer = Dense(units=10, activation='softmax')

model = Sequential([input_layer,
                    hidden_layer,
                    output_layer])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])