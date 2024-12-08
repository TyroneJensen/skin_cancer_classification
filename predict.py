import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
import gradio as gr

# Load the trained model
model = load_model('skin_cancer_model.h5')

# Function to load and preprocess image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Predict function
def predict_image(img_path):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    return 'Malignant' if prediction[0][0] > 0.5 else 'Benign'

# Define Gradio interface
def gradio_interface(img_path):
    result = predict_image(img_path)
    return result

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.inputs.Image(type="filepath"),
    outputs="text",
    title="Skin Cancer Classification",
    description="Upload a skin lesion image to classify it as Benign or Malignant."
)

# Launch the interface
iface.launch()

# Example usage
img_path = 'C:\\Users\Arthur\CODE\DataProjects\skin_cancer_classification\data\organized_test\malignant\ISIC_0000165.jpg'  # Replace with path to an image
result = predict_image(img_path)
print(f'The image is classified as: {result}')
