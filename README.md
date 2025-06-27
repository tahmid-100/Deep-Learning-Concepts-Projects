# 🐕🐱 Dog vs Cat Classifier

A deep learning project that classifies images as either dogs or cats using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The project includes a web application interface built with Streamlit for easy image classification.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Model Performance](#model-performance)
- [Web Application](#web-application)

## 🎯 Project Overview

This binary image classifier uses a CNN architecture to distinguish between dog and cat images with high accuracy. The model is trained on a dataset of 20,000 images (100x100 pixels, RGB) and deployed through an interactive web interface.

**Key Features:**
- ✅ Binary image classification (Dog vs Cat)
- ✅ CNN architecture with 875,777 parameters
- ✅ Interactive Streamlit web application
- ✅ Real-time image prediction
- ✅ Dropout regularization to prevent overfitting

## 📊 Dataset

- **Training Images**: 20,000 images
- **Image Dimensions**: 100x100 pixels
- **Color Channels**: 3 (RGB)
- **Classes**: Binary classification (0 = cat, 1 = dog)
- **Train/Test Split**: 80/20 split with random state 42

## 🏗️ Model Architecture

The CNN model consists of **875,777 trainable parameters** organized in the following layers:

### Convolutional Layers
| Layer | Filters | Kernel Size | Output Shape | Parameters |
|-------|---------|-------------|--------------|------------|
| Conv2D 1 | 32 | 3x3 | (98, 98, 32) | 896 |
| MaxPool 1 | - | 2x2 | (49, 49, 32) | 0 |
| Conv2D 2 | 64 | 3x3 | (47, 47, 64) | 18,496 |
| MaxPool 2 | - | 2x2 | (23, 23, 64) | 0 |
| Conv2D 3 | 64 | 3x3 | (21, 21, 64) | 36,928 |
| MaxPool 3 | - | 2x2 | (10, 10, 64) | 0 |

### Fully Connected Layers
- **Flatten Layer**: Converts 2D feature maps to 1D vector
- **Dense Layer 1**: 128 neurons, ReLU activation
- **Dropout Layer**: 50% dropout rate for regularization
- **Output Layer**: 1 neuron, Sigmoid activation (binary classification)

## 🌐 Web Application

The Streamlit web application provides an intuitive interface with the following features:

- **📤 File Upload**: Drag and drop or browse for image files
- **🔍 Real-time Prediction**: Instant classification results
- **📊 Confidence Score**: Probability scores for predictions
- **🖼️ Image Preview**: Display uploaded images
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🖼️ Demo Screenshot

![Sample Image](Computer%20Vision/Cat%20vs%20dog/1.png)

![Sample Image](Computer%20Vision/Cat%20vs%20dog/2.png)

![Sample Image](Computer%20Vision/Cat%20vs%20dog/3.png)

![Sample Image](Computer%20Vision/Cat%20vs%20dog/4.png)



