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

The dual-channel model combines features extracted from both **InceptionV3** and **Xception** architectures. Each model processes the MRI images separately, and their outputs are combined in the final layers for accurate classification.

This approach leverages the unique feature extraction capabilities of both models, aiming to maximize accuracy by capturing more detailed patterns in the MRI images, especially useful in medical imaging.

## Web Application Features

The web application allows users to upload MRI images for classification. It features:
- **User Interface**: An intuitive, responsive UI for image upload and result visualization.
- **Image Processing**: Upon image upload, the model processes the MRI and returns the tumor type or indicates if no tumor is detected.
- **Results Display**: The prediction results, including tumor type (if detected), are presented on the same page.

### Screenshots

![Home Page](Home.png)

The screenshot above shows the main interface of the web application, where users can upload MRI images and view the results.
