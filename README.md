# ♻️ Garbage Detection with AI

This repository contains the code and resources for the **Garbage Detection with AI Project**, where we explore how to build, train, and deploy an object detection model for classifying waste materials using deep learning.

The project is based on a custom YOLOv11 model trained to detect the following waste categories:

- **Biological**
- **Cardboard**
- **Glass**
- **Metal**
- **Paper**
- **Plastic**
- **Trash**

---

## 📂 Dataset
We used a custom dataset for this project, which can be accessed here:  
👉 [Download Garbage Dataset (Google Drive)](https://drive.google.com/drive/u/1/folders/1CZQoOW_K0V55wY8tJHGRiSaY-DjPgbw6)

---

## 📚 Topics Covered in the Workshop
1. **Data Annotation**  
   - Understanding the importance of labeling for object detection  
   - Annotating images using tools like Makesense.ai, Roboflow or LabelImg  

2. **Exploratory Data Analysis (EDA)**  
   - Visualizing images and class distribution  
   - Understanding dataset balance  

3. **Data Augmentation**  
   - Applying transformations to increase dataset diversity  
   - Using YOLO-compatible augmentation techniques  

4. **Model Training**  
   - Training YOLOv11 on custom dataset  
   - Monitoring metrics like precision, recall, and mAP  

5. **Running Inference**  
   - Testing trained model on new images and videos  
   - Evaluating real-time performance  

6. **Building App for Deployment using Gradio**  
   - Creating a simple Gradio app with tabs: **Image Inference**, **Video Inference**, and **Real-time Inference**  
   - Deploying the app for interactive testing  

---

## ⚙️ Installation

Clone this repository and install the required dependencies.

```bash
git clone https://github.com/abuelgasimsaadeldin/garbage-project.git
cd garbage-detection
