import gradio as gr
import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Classes from your training
classes = ['biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

sample_images = [
    "sample_images/sample1.jpg",
    "sample_images/sample2.jpg",
    "sample_images/sample3.jpg",
]

# ----------- IMAGE INFERENCE -----------
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


# ----------- VIDEO INFERENCE -----------
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        # Return RGB frame for Gradio
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()

# ----------- GRADIO UI -----------
image_demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Detected Image"),
    title="Waste Object Detection - Image",
    examples=sample_images,
    allow_flagging="never"
)

video_demo = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Image(type="numpy", label="Detected Video Frames"),
    title="Waste Object Detection - Video",
    allow_flagging="never"
)

# Combine into tabs
demo = gr.TabbedInterface(
    [image_demo, video_demo],
    tab_names=["Image Inference", "Video Inference"]
)

demo.launch(share=True)
