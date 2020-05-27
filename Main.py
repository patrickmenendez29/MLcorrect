import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from pylab import *

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-2.0,-1.0, 0.0, 1.0, 2.0, 3.0, 4.0,5.0], dtype=float)
ys = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0,4.0,5.0], dtype=float)



if __name__ == "__main__":
    model.fit(xs, ys, epochs=100)
    print(round(float(model.predict([100]))))



