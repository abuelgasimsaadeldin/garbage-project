import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")
classes = ['biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Function to perform image inference
def predict_image(image_path):
    image = cv2.imread(image_path)
    results = model.predict(source=image_path)
    output = results[0]

    for box, conf, cls in zip(output.boxes.xyxy, output.boxes.conf, output.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_name = classes[int(cls)]
        color = (0, 255, 0)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label with class + confidence
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to display the image in the Tkinter window
def show_image(image_path):
    image = predict_image(image_path)
    image_pil = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image_pil)

    # Update the Tkinter label with the processed image
    panel.config(image=image_tk)
    panel.image = image_tk

# Function to open a file dialog for image selection
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        show_image(file_path)

# Function to process video frames
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")

    def process_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return

        results = model.predict(source=frame, verbose=False)
        output = results[0]

        for box, conf, cls in zip(output.boxes.xyxy, output.boxes.conf, output.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_name = classes[int(cls)]
            color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Convert frame to RGB and display in Tkinter window
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(frame_pil)

        # Update Tkinter label with the current frame
        panel.config(image=frame_tk)
        panel.image = frame_tk

        # Schedule next frame update
        panel.after(30, process_frame)

    process_frame()

# Function to open a file dialog for video selection
def upload_video():
    file_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        predict_video(file_path)

# Setting up Tkinter window
root = tk.Tk()
root.title("Waste Object Detection")
root.geometry("800x600")

# Create an image display panel
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# Create buttons to upload image and video
btn_image = tk.Button(root, text="Upload Image", command=upload_image)
btn_image.pack(side="left", padx=20, pady=20)

btn_video = tk.Button(root, text="Upload Video", command=upload_video)
btn_video.pack(side="right", padx=20, pady=20)

# Start the Tkinter event loop
root.mainloop()
