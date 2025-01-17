# Traffic Sign, Lane, and Pedestrian Detection System

## Project Overview

This project focuses on developing a comprehensive system for detecting traffic signs, lanes, and pedestrians. The project leverages deep learning techniques, computer vision, and image processing to create a robust detection system. The application uses a Convolutional Neural Network (CNN) model for traffic sign detection, Hough Transform for lane and pedestrian detection, and integrates various Python libraries for data processing and visualization.

## Features

Traffic Sign Detection: Built using a CNN model to classify and detect various traffic signs.

Lane Detection: Utilizes Hough Transform for accurate lane detection in real-time.

Pedestrian Detection: Employs image processing techniques and Hough Transform to identify pedestrians.

Data Augmentation: Implements image preprocessing and augmentation for robust training.

## Libraries and Tools Used

### Core Libraries:

numpy for numerical computations

matplotlib.pyplot for data visualization

pandas for data manipulation

###Deep Learning:

#### keras and its submodules:

Sequential

Dense

Dropout, Flatten

Conv2D, MaxPooling2D

#### tensorflow with Adam optimizer

### Image Processing:

cv2 (OpenCV) for image operations

sklearn.model_selection for splitting datasets

ImageDataGenerator for augmenting training images

### Data Management:

pickle and joblib for saving and loading models

os for directory and file management

random for random operations


## How to Run the Project

## Prerequisites

Install Python 3.8 or higher.

Install required dependencies:

pip install -r requirements.txt

### Steps to Execute

Clone the repository:

git clone https://github.com/your_username/traffic-detection-system.git

Navigate to the project directory:

cd traffic-detection-system

Prepare the dataset and place it in the data directory.

Train the traffic sign detection model:

python src/traffic_sign_detection.py

Run the lane detection system:

python src/lane_detection.py

Execute the pedestrian detection system:

python src/pedestrian_detection.py

## Methodology

### Traffic Sign Detection

Architecture: The CNN model includes convolutional layers with max pooling and dropout for regularization, followed by dense layers for classification.

Training: Used Adam optimizer and categorical cross-entropy loss.

Augmentation: Applied transformations using ImageDataGenerator for better generalization.

### Lane Detection

Algorithm: Utilized the Hough Transform to detect lane lines from the input video stream or images.

Preprocessing: Applied Gaussian blur, edge detection, and region masking for accurate results.

### Pedestrian Detection

Technique: Leveraged OpenCV and Hough Transform for detecting pedestrian shapes in the frame.

Enhancements: Applied bounding boxes to highlight detected pedestrians.

## Dataset

Traffic Sign Dataset: Contains labeled images of traffic signs for model training and validation.

Lane and Pedestrian Dataset: Custom or publicly available datasets processed for edge detection and object identification.

## Results

Traffic Sign Detection: Achieved XX% accuracy on the test dataset.

Lane Detection: Successfully identified lanes in various driving conditions.

Pedestrian Detection: Detected pedestrians with high precision and recall.

## Future Enhancements

Integrate the system into a real-time application using video feeds.

Improve pedestrian detection with advanced object detection algorithms like YOLO or SSD.

Optimize the traffic sign detection model for deployment on edge devices.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgments

OpenCV and TensorFlow communities for excellent documentation and tutorials.

Kaggle and other data sources for datasets.

## Contact

For questions or collaboration opportunities, please contact aryashubham2015@gmail.com.

