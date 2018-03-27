import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
from keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
test_sample=cv2.imread("data/validation/bad/l_0_id_470.jpg")
test_sample2 = np.expand_dims(test_sample, axis=0)
out=model.predict(test_sample2, batch_size=None, verbose=0, steps=1)
print(out)