

import cv2
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from ultralytics import YOLO

# Load pretrained YOLO model
print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

# Open image selection window
root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select a Traffic Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png")
    ]
)

root.destroy()

if not image_path:
    print("No image selected!")
    exit()

# Perform object detection
print("Detecting objects...")

results = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640
)

# Draw bounding boxes and labels
result = results[0]
output_image = result.plot()

# Create output folder
output_folder = Path(__file__).resolve().parent / "output"
output_folder.mkdir(exist_ok=True)

# Save detected image
output_path = output_folder / "detected_output.jpg"

cv2.imwrite(str(output_path), output_image)

# Print detected objects
print("\nDetected Objects:")

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    label = model.names[class_id]

    print(f"{label}: {confidence:.2f}")

print("\nDetection completed!")
print("Output saved at:", output_path)

# Display output directly from PyCharm
cv2.imshow("Visionport - Object Detection", output_image)

print("\nPress any key on the image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()
