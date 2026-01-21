# 13. Dog vs. Cat Classifier with CNN (Convolutional Neural Network)



# libraries
import tensorflow as tf
from keras import layers, models
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import json


# url: https://www.kaggle.com/datasets/tongpython/cat-and-dog
# Define paths to the dataset (update these paths with the actual dataset location)
train_dir = 'dataset/cat_dog/training_set'
validation_dir = 'dataset/cat_dog/test_set'


# Define ImageDataGenerators for data augmentation and rescaling
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
  rescale=1./255,   # rescale pixel values (0-255) to (0-1)
  rotation_range=40,   # randomly rotate images
  width_shift_range=0.2,   # randomly shift images horizontally
  height_shift_range=0.2,   # randomly shift images vertically
  shear_range=0.2,   # randomly shear images
  zoom_range=0.2,   # randomly zoom in on images
  horizontal_flip=True,   # randomly flip images horizontally
  fill_mode='nearest'   # fill pixels that may have been lost after transformation
)


# For the validation data, we just rescale (no data augmentation)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# Load training and validation data
train_generator = train_datagen.flow_from_directory(
  train_dir,
  target_size=(150, 150),   # resize all images to 150x150
  batch_size=32,
  class_mode='binary'   # binary classification (Dog or Cat)
)

validation_generator = validation_datagen.flow_from_directory(
  validation_dir,
  target_size=(150, 150),   # resize all images to 150x150
  batch_size=32,
  class_mode='binary'   # binary classification (Dog or Cat)
)


# Define the CNN model
model = models.Sequential()

# First convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output from the convolutional layers and add fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))   # outut layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print summary of the model
model.summary()

# Train the model
history = model.fit(
  train_generator,
  steps_per_epoch=100,   # number of batches per epoch
  epochs=20,   # number of epoch to train
  validation_data=validation_generator,
  validation_steps=50   # number of batches for validation
)


# Save the model with class names
class_names = list(train_generator.class_indices.keys())

with open('saved_models/class_names.json', 'w') as f:
  json.dump(class_names, f)

model.save('saved_models/cat_dog_model.keras')


# Final Accuracy from training history
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {final_train_acc * 100:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%")

# Proper evaluation on validation/test set
test_loss, test_accuracy = model.evaluate(validation_generator)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# Plot training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


# Test the model with new image
def predict_image(model, img_path):
  img = image.load_img(img_path, target_size=(150, 150))   # load the image
  img_array = image.img_to_array(img)   # convert image to array
  img_array = np.expand_dims(img_array, axis=0)   # add batch dimension
  img_array /= 255.0   # normalize the image (rescale pixel values to [0, 1])
  
  prediction = model.predict(img_array)   # make prediction
  
  if prediction[0] > 0.5:
    print(f'The image is predicted to be a Dog with a confidence of {prediction[0][0]:.2f}')
  else:
    print(f'The image is predicted to be a Cat with a confidence of {1- prediction[0][0]:.2f}')

# Example: Test the classifier with a new image
predict_image(model, 'test_set/cat_dog/cat1.jpg')