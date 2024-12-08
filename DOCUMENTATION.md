# Detailed Documentation for Skin Cancer Image Classification Project

## Project Overview
This project aims to classify skin cancer images using a Convolutional Neural Network (CNN). The model is trained to differentiate between benign and malignant skin lesions using the ISIC Skin Cancer Dataset.

## Dataset
- **Source**: ISIC Skin Cancer Dataset
- **Format**: JPG images with accompanying metadata
- **Metadata**: Contains image IDs and labels (benign or malignant)
- **Directory Structure**:
  - `data/train/`: Contains training images
  - `data/test/`: Contains testing images
  - `data/train/ISIC_images_metadata/metadata.csv`: Metadata for training images
  - `data/test/ISIC_images_metadata/metadata.csv`: Metadata for testing images

## Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Layers**:
  - Convolutional layers with ReLU activation
  - Max pooling layers
  - Flatten layer
  - Dense layers with dropout
  - Output layer with sigmoid activation for binary classification

## Updated Model Architecture
- Added more convolutional layers with increased filters to capture complex features.
- Introduced batch normalization after each convolutional layer to stabilize and accelerate training.
- Added an additional dense layer to enhance the model's learning capacity.

## Preprocessing Steps
- Resize images to 128x128 pixels
- Normalize pixel values by dividing by 255

## Training Configuration
- **Batch Size**: 32
- **Epochs**: 20
- **Optimizer**: Adam
- **Loss Function**: Binary cross-entropy
- **Learning Rate**: 0.0001
- **Learning Rate Scheduler**: Reduces the learning rate by 10% after 5 epochs

## Hyperparameter Tuning
- Adjusted the learning rate to 0.0001 for better convergence.
- Increased the number of epochs to 20 to allow more training time.
- Implemented a learning rate scheduler that reduces the learning rate by 10% after 5 epochs.

## Data Handling
### Initial Data Format
- The ISIC Skin Cancer Dataset is provided as a collection of JPG images, each representing a skin lesion.
- Accompanying each image is a metadata file (`metadata.csv`) that contains information about the image, including:
  - `isic_id`: Unique identifier for each image.
  - `benign_malignant`: Label indicating whether the lesion is benign or malignant.

### Data Processing Steps
1. **Download and Organize Data**:
   - Download the dataset and place the images in the `data/train` and `data/test` directories.
   - Ensure that the metadata files are placed in `data/train/ISIC_images_metadata` and `data/test/ISIC_images_metadata`.

2. **Reading Metadata**:
   - The metadata is read using Pandas to extract the `isic_id` and `benign_malignant` columns.
   - These columns are essential for organizing the images and labeling them for training.

3. **Image Organization**:
   - Images are organized into `organized_train` and `organized_test` directories based on their labels.
   - The script iterates over the metadata, and for each entry, it checks if the `isic_id` and `benign_malignant` values are valid.
   - Valid images are copied to their respective directories (`benign` or `malignant`) within `organized_train` and `organized_test`.

4. **Handling Missing Data**:
   - The script includes checks to skip entries with missing or malformed `isic_id` or `benign_malignant` values to prevent errors during processing.

5. **Data Generators**:
   - Once organized, the images are fed into Keras' `ImageDataGenerator` for preprocessing and augmentation.
   - This step includes resizing images to 128x128 pixels and normalizing pixel values by dividing by 255.

This structured data handling ensures that the images are correctly prepared and labeled for training the CNN model.

## Running the Model
1. Ensure the dataset is downloaded and placed in the `data/` directory.
2. Run `model.py` to organize images and train the model.
3. Use `predict.py` to make predictions on new images.

## Gradio Interface
- A Gradio interface has been implemented in the `predict.py` script.
- Users can upload a skin lesion image through the interface to classify it as Benign or Malignant.
- The interface provides a user-friendly way to interact with the model and view results instantly.

### How to Use
1. Run the `predict.py` script.
2. The Gradio interface will launch in your default web browser.
3. Upload an image of a skin lesion to receive a classification result.

## Future Improvements
- Implement data augmentation techniques to enhance model performance.
- Explore transfer learning with pre-trained models like VGG16 or ResNet.
- Add more comprehensive error handling and logging.

## Security Considerations
- Ensure proper handling of medical image data.
- Implement data anonymization if required.
- Be mindful of patient privacy regulations.
