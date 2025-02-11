import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(gpu)

# Data Preprocessing
data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'JPG', 'JPEG'] #These are the allowed image types

# Debugging: Count the number of valid images per class before filtering
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    if os.path.isdir(class_path):
        valid_images = [img for img in os.listdir(class_path) if imghdr.what(os.path.join(class_path, img)) in image_exts]
        print(f"Class {image_class} has {len(valid_images)} valid images before filtering.")

# Image filtering and removal
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    if os.path.isdir(class_path):  # Ensure it's a directory
        for image in os.listdir(class_path):
            image_path = os.path.join(class_path, image)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Failed to load image: {image_path}")
                    os.remove(image_path)
                    continue
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print(f"Image not in ext list: {image_path}")
                    os.remove(image_path)
            except Exception as e:
                print(f"Issue with image: {image_path}")
                print(e)
                os.remove(image_path)

# Check if images remain after filtering
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    if os.path.isdir(class_path):  # Ensure it's a directory
        remaining_images = [img for img in os.listdir(class_path) if imghdr.what(os.path.join(class_path, img)) in image_exts]
        print(f"Class {image_class} has {len(remaining_images)} images remaining after filtering.")

# Load and Resize Dataset
data = tf.keras.utils.image_dataset_from_directory(
    'data', image_size=(256, 256), batch_size=8
)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Plot Sample Images
fig, ax = plt.subplots(ncols=9, figsize=(20, 20))
for idx, img in enumerate(batch[0][:9]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# Normalize Data
data = data.map(lambda x, y: (x / 255, y))

# Split Dataset
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2) + 1
test_size = int(len(data) * 0.1) + 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Model Architecture
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Dropout(0.5),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Dropout(0.5),

    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(),
    Dropout(0.5),

    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(9, activation='softmax')  # 9 CLasses total
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Training the Model
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=15, validation_data=val, callbacks=[tensorboard_callback])

# Plotting the Loss
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='blue', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

model.save("twilight_model.h5") #Save model to use in run.py