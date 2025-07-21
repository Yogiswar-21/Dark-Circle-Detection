## Project Title
Smart Eye Care: A Deep Learning Model for Real-Time Dark Circle Detection

**Submitted By**  
[M.SAI KUMAR (R200213)]
[K.VENKATA YOGISWARA REDDY (R200215)]

## Description

Dark circles under the eyes are a common cosmetic and dermatological concern often associated with fatigue, aging, and various underlying health conditions. With increasing demand for automated skincare solutions, the need for accurate detection of dark circles using computer vision and deep learning has grown significantly. This project presents a novel approach to detecting dark circles from facial images using object detection models such as YOLOv8 .

The study utilizes a custom dataset prepared specifically for this project. Facial images were collected and annotated using the Roboflow, which enabled precise detection and mapping of facial landmarks, including the periorbital region. This approach ensured that the dataset is tailored to the requirements of dark circle detection, providing high-quality, relevant samples for training and evaluating the model’s performance. By applying data preprocessing, and training deep learning models on Google Colab, the system aims to accurately localize and classify
regions exhibiting dark circles.

In addition to implementing the detection system, this project conducts a comparative analysis of various research works from IEEE Xplore, Pubmed to validate the relevance and novelty of the proposed approach. Evaluation metrics such as mean Average Precision (mAP), precision, and recall are used to assess model performance. The expected outcome is a lightweight, real-
time applicable dark circle detection tool that can be integrated into skincare diagnostics, mobile apps, or dermatological support systems.

## Technologies Used

**Google Colab (Free GPU)**
**Ultralytics YOLOv8 Library**
**OpenCV + Python**

## Installation Steps

**importing dataset**
from google.colab import drive
drive.mount('/content/drive')
**libraries**
!pip install ultralytics
!pip install gradio
from ultralytics import YOLO

## How to Run the Project

**model training**
    model = YOLO('yolov8n.pt')  # YOLOv8 nano model
**Train the model on your custom dataset**
    model.train(data='/content/drive/MyDrive/Colab Notebooks/Dark Circles Detection.v1i.yolov8/data.yaml', epochs=250, imgsz=640)
**downloading weights file**
    from google.colab import files
    files.download('/content/runs/detect/train/weights/best.pt')
**predicting the model**
    results = model.predict('/content/drive/MyDrive/Colab Notebooks/sdc.png', conf=0.5)
**visualization**
    import gradio as gr
    from ultralytics import YOLO
    import datetime

**Load the YOLO model**
    model = YOLO('runs/detect/train/weights/best.pt')

## Project Flow

1. Dataset Collection
Collect facial images representing various individuals, ages, and ethnicities to ensure diversity in the dataset.

Use Roboflow to annotate the dataset by labeling the periorbital region (around the eyes), focusing on dark circles.

2. Image Annotation
Use Roboflow or a similar tool to annotate the dataset. Label regions under the eyes, marking dark circles and other relevant facial features.

3. Preprocessing
Resize the images to a consistent dimension (e.g., 640x640).

Normalize pixel values and apply data augmentation techniques (rotation, flipping, etc.) to increase dataset robustness.

4. Model Training
Use the YOLOv8 model to train on the preprocessed dataset. Experiment with different hyperparameters (e.g., learning rate, batch size, number of epochs) to achieve the best results.

5. Evaluation
Evaluate the model using metrics like Mean Average Precision (mAP), precision and recall.

6. Visualization of Results
Use OpenCV and Gradio to visualize detection results on test images. Display bounding boxes around the dark circles detected.

