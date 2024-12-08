import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import shutil
from sklearn.metrics import classification_report, confusion_matrix

# Constants
data_dir = 'data/'
img_height, img_width = 128, 128
batch_size = 32
epochs = 20

# Define paths for metadata
train_metadata_path = os.path.join(data_dir, 'train', 'ISIC_images_metadata', 'metadata.csv')
test_metadata_path = os.path.join(data_dir, 'test', 'ISIC_images_metadata', 'metadata.csv')

# Load metadata
train_metadata = pd.read_csv(train_metadata_path)
test_metadata = pd.read_csv(test_metadata_path)

# Create directories for organized data
organized_train_dir = os.path.join(data_dir, 'organized_train')
organized_test_dir = os.path.join(data_dir, 'organized_test')
os.makedirs(organized_train_dir, exist_ok=True)
os.makedirs(organized_test_dir, exist_ok=True)

# Organize train images based on metadata
for index, row in train_metadata.iterrows():
    image_id = row['isic_id']
    label = row['benign_malignant']
    if pd.isna(image_id) or pd.isna(label):
        continue
    src_path = os.path.join(data_dir, 'train', f'{image_id}.jpg')
    dst_dir = os.path.join(organized_train_dir, label)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src_path, dst_dir)

# Organize test images based on metadata
for index, row in test_metadata.iterrows():
    image_id = row['isic_id']
    label = row['benign_malignant']
    if pd.isna(image_id) or pd.isna(label):
        continue
    src_path = os.path.join(data_dir, 'test', f'{image_id}.jpg')
    dst_dir = os.path.join(organized_test_dir, label)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src_path, dst_dir)

# Data preparation
datagen = ImageDataGenerator(rescale=1./255)

# Update data generators to use organized directories
train_generator = datagen.flow_from_directory(
    organized_train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    organized_test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Updated Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile with adjusted learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch > 5:
        lr = lr * 0.9
    return lr

# Model training with learning rate scheduler
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[LearningRateScheduler(lr_schedule)]
)

# Evaluate the model on the validation data and print the results
predictions = model.predict(validation_generator)
predicted_classes = (predictions > 0.5).astype("int32")
true_classes = validation_generator.classes

# Print classification report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

# Save the model
model.save('skin_cancer_model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
