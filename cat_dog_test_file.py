import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import json

filepath = 'saved_models/cat_dog_model.keras'

loaded_model = load_model(filepath)

with open('saved_models/class_names.json', 'r') as f:
  class_names = json.load(f)

print(f'Class names: {class_names}')

# print(loaded_model)

# loaded_model.summary()


# Test the model with new image
def predict_image(model, img_path):
  img = image.load_img(img_path, target_size=(150, 150))   # load the image
  img_array = image.img_to_array(img)   # convert image to array
  img_array = np.expand_dims(img_array, axis=0)   # add batch dimension
  img_array /= 255.0   # normalize the image (rescale pixel values to [0, 1])
  
  prediction = model.predict(img_array)[0][0]   # make prediction
  
  if prediction > 0.5:
    label = class_names[1]
    confidence = prediction
  else:
    label = class_names[0]
    confidence = 1 - prediction

  print(f"Predicted: {label} with (Confidence: {confidence * 100:.2f}%)")


# Example: Test the classifier with a new image
predict_image(loaded_model, 'test_set/cat_dog/dog3.jpeg')