import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

# Load the training and testing data
train_dir = 'data/train'
test_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

# Load the VGG16 model
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in vgg.layers:
    layer.trainable = False

# Add new layers on top of the pre-trained layers
x = tf.keras.layers.Flatten()(vgg.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Create the new model
model = tf.keras.models.Model(inputs=vgg.input, outputs=x)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=10, validation_data=test_data)

import matplotlib.pyplot as plt

# Plot the accuracy and loss over time
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Evaluate the model on the testing data
test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)

import numpy as np
from tensorflow.keras.preprocessing import image

# Load an image for testing
test_image = image.load_img('data/cat.jpg', target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make a prediction on the test image
prediction = model.predict(test_image)
if prediction[0][0] == 1:
    print('Dog')
else:
    print('Cat')
