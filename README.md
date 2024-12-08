# Skin Cancer Image Classification Project

## Overview
This project focuses on classifying skin cancer images using deep learning techniques. We will utilize a convolutional neural network (CNN) to identify different types of skin lesions from images.

## Dataset
- **Source**: ISIC Skin Cancer Dataset
- **Content**: Contains images of various skin lesions labeled with their respective types.

## Project Structure
- `data/`: Directory to store dataset images.
- `model.py`: Script to build and train the CNN model.
- `predict.py`: Script to make predictions on new images and host the Gradio interface.
- `requirements.txt`: List of project dependencies.

## Documentation
For detailed information about the project, refer to the `DOCUMENTATION.md` file.

## Setup Instructions
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Download the ISIC Skin Cancer Dataset and place it in the `data/` directory.
4. Run `model.py` to train the CNN model.
5. Use `predict.py` to classify new skin lesion images.

## Recent Updates
- Added checks for missing metadata values to prevent errors during image organization.
- Created `DOCUMENTATION.md` for detailed project information.

## Recent Model Updates
- Enhanced model architecture with additional convolutional layers and batch normalization.
- Adjusted learning rate and epochs for improved training.
- Implemented a learning rate scheduler to dynamically adjust learning rate during training.

These updates are designed to enhance the model's accuracy and robustness.

## Gradio Interface
- A user-friendly Gradio interface has been added to the `predict.py` script.
- Allows users to upload images and receive classification results directly through a web interface.

### Usage Instructions
1. Run the `predict.py` script to start the interface.
2. Use the interface to upload a skin lesion image and get a prediction.

These updates make it easier for users to interact with the model and obtain results quickly.

## Future Work
- Implement data augmentation techniques to improve model robustness.
- Explore transfer learning with pre-trained models like VGG16 or ResNet.
