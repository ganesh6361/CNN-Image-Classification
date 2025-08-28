CNN - MNIST Image Classifier ✍️
This project demonstrates a fundamental machine learning workflow using a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The goal is to build a model that can accurately identify a digit (0-9) from an image. This serves as a classic "Hello, World" for deep learning and computer vision.

Machine Learning Workflow
The project follows a standard machine learning pipeline:

Data Preparation: The raw image data is loaded from the MNIST dataset, normalized to a range of [0, 1], and reshaped to a format suitable for a CNN.

Model Architecture: A simple yet effective CNN is built using TensorFlow and Keras, consisting of convolutional, pooling, flattening, and dense layers.

Model Training: The model is trained on the training data using an adam optimizer and a sparse_categorical_crossentropy loss function. To prevent overfitting and save resources, the training utilizes Early Stopping and Model Checkpointing callbacks.

Model Evaluation: The trained model's performance is evaluated on a held-out test set to ensure its accuracy on unseen data.

Prediction: The final model is used to make a prediction on a new, single image.

Getting Started
To run this project, you will need to have Python and the necessary libraries installed.

1. Install Libraries
You can install the required libraries using pip:

pip install tensorflow matplotlib numpy

2. Run the Notebook
Open the google Colab file (your-notebook-name.ipynb) in your preferred environment and run the cells in sequence.

Results
The trained CNN achieves high accuracy in classifying handwritten digits. The model's performance on the test set is an excellent indicator of its ability to generalize to new, unseen images.

The following is an example of a successful prediction from the model.

Technologies Used
Python: The core programming language for the project.

TensorFlow & Keras: The primary deep learning framework used to build and train the CNN.

NumPy: Used for efficient numerical operations and data manipulation.

Matplotlib: Used for data visualization and displaying the predicted image.

Google Colab: The environment where the code was written and executed.
