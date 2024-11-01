# Brain Tumor Detection Using Dual-Channel Deep Learning Models

## Overview

This project implements a dual-channel deep learning model to classify brain tumors in MRI images. The model combines **InceptionV3** and **Xception** pretrained models to explore whether a single model or dual-channel architecture yields better classification performance. The project classifies MRI images into one of three tumor types:

- **Glioma**
- **Pituitary**
- **Meningioma**
- **No Tumor**

The final model achieved an impressive **98% accuracy**, demonstrating its robustness in classifying MRI images. This research was also published in the **Evolutionary Artificial Intelligence Proceedings of ICEAI 2023**. [Read the paper here](https://link.springer.com/chapter/10.1007/978-981-99-8438-1_8).

## Project Highlights

- **Dual-Channel Model Architecture**: Integrates both InceptionV3 and Xception models, leveraging each model's strengths to improve accuracy in predicting tumor types.
- **Extensive Model Testing**: Experimented with multiple architectures to determine the optimal setup for maximum accuracy.
- **End-to-End Web Application**: Developed a web interface that allows users to upload MRI images and view predictions on tumor type. This application is useful for medical professionals seeking diagnostic support.

## Model Structure

The dual-channel model architecture combines feature extraction from two powerful pretrained models, **InceptionV3** and **Xception**, to classify MRI images into four classes: **glioma**, **meningioma**, **no tumor**, and **pituitary**.

### Architecture Details

- **Input Layer**: The model accepts input MRI images of shape `(224, 224, 3)`.
  
- **First Channel - InceptionV3**: 
  - The InceptionV3 model, pretrained on ImageNet, processes the input image.
  - The final output of this channel is passed through a **Global Average Pooling** layer to reduce spatial dimensions.

- **Second Channel - Xception**: 
  - The Xception model, also pretrained on ImageNet, processes the same input image.
  - Similar to the first channel, the output is passed through a **Global Average Pooling** layer.

- **Merging and Final Layers**:
  - The outputs from both InceptionV3 and Xception channels are concatenated, combining their extracted features.
  - The merged output is passed through a fully connected **Dense** layer with 256 units and ReLU activation for feature refinement.
  - Finally, a **Dense output layer** with a softmax activation provides probabilities for the four classes: glioma, meningioma, no tumor, and pituitary.

### Summary

This dual-channel architecture leverages both InceptionV3 and Xception for feature extraction, providing high classification accuracy by capturing complementary features from both models.


## Web Application Features

The web application allows users to upload MRI images for classification. It features:
- **User Interface**: An intuitive, responsive UI for image upload and result visualization.
- **Image Processing**: Upon image upload, the model processes the MRI and returns the tumor type or indicates if no tumor is detected.
- **Results Display**: The prediction results, including tumor type (if detected), are presented on the same page.

### Screenshots

![Home Page](Home.png)

The screenshot above shows the main interface of the web application, where users can upload MRI images and view the results.
