
import os
import json
import time
import statistics
import numpy as np
import torch
import psutil
import pandas as pd

from pathlib import Path
from PIL import Image
from ultralytics import YOLO


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "yolov8n.pt"
IDD_DIR = BASE_DIR / "data" / "IDD_DATASET"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("VISIONPORT IDD BENCHMARK")
print("=" * 50)

print("Model path:", MODEL_PATH)
print("Dataset path:", IDD_DIR)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLOv8 model not found: {MODEL_PATH}"
    )

if not IDD_DIR.exists():
    raise FileNotFoundError(
        f"IDD dataset folder not found: {IDD_DIR}"
    )


# ==========================================
# 2. FIND AND VALIDATE IMAGES
# ==========================================

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

all_images = [
    p for p in IDD_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
]

valid_images = []
invalid_images = []

print("\nChecking IDD images...")

for image_path in all_images:

    try:
        with Image.open(image_path) as img:
            img.verify()

        valid_images.append(image_path)

    except Exception as e:

        invalid_images.append({
            "file": str(image_path),
            "error": str(e)
        })

print("Total images:", len(all_images))
print("Valid images:", len(valid_images))
print("Invalid images:", len(invalid_images))

# Sort images numerically: Image_0, Image_1, Image_2...
def image_number(path):
    try:
        return int(path.stem.replace("Image_", ""))
    except ValueError:
        return float("inf")


valid_images.sort(key=image_number)

# Check dataset size
if len(valid_images) < 200:
    raise ValueError(
        f"Expected at least 200 valid images, "
        f"but found only {len(valid_images)}."
    )

# Select exactly 200 images
benchmark_images = valid_images[:200]

print(
    "\nImages selected for benchmark:",
    len(benchmark_images)
)

print(
    "First image:",
    benchmark_images[0].name
)

print(
    "Last image:",
    benchmark_images[-1].name
)


# ==========================================
# 3. CHECK NVIDIA GPU
# ==========================================

print("\nChecking GPU...")

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():

    DEVICE = 0

    print("GPU available: True")
    print("GPU name:", torch.cuda.get_device_name(0))

else:

    DEVICE = "cpu"

    print("GPU available: False")
    print("Running on CPU instead.")


# ==========================================
# 4. LOAD YOLOV8 MODEL
# ==========================================

print("\nLoading YOLOv8 model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")

print("\nModel classes:")

for class_id, class_name in model.names.items():
    print(class_id, ":", class_name)


# ==========================================
# 5. MODEL WARM-UP
# ==========================================

print("\nStarting model warm-up...")

WARMUP_RUNS = 10

for i in range(WARMUP_RUNS):

    image_path = benchmark_images[
        i % len(benchmark_images)
    ]

    model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        device=DEVICE,
        verbose=False
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(
        f"Warm-up: {i + 1}/{WARMUP_RUNS}"
    )

print("Warm-up completed!")


# ==========================================
# 6. BENCHMARK SETTINGS
# ==========================================

IMAGE_SIZE = 640
CONFIDENCE = 0.25
MEASUREMENT_ROUNDS = 3

latencies = []
prediction_records = []

process = psutil.Process(os.getpid())

ram_before = (
    process.memory_info().rss / (1024 ** 2)
)

if torch.cuda.is_available():

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


# ==========================================
# 7. RUN BENCHMARK
# ==========================================

print("\nStarting VisionPort benchmark...")

total_start = time.perf_counter()

for round_no in range(MEASUREMENT_ROUNDS):

    print(
        "\nMeasurement round:",
        round_no + 1
    )

    for image_index, image_path in enumerate(
        benchmark_images
    ):

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        latencies.append(latency_ms)

        if not results:
            continue

        result = results[0]
        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0].item())

                confidence = float(
                    box.conf[0].item()
                )

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": confidence,
                    "bbox_xyxy": xyxy
                })

        # Save predictions only in final round
        if round_no == MEASUREMENT_ROUNDS - 1:

            prediction_records.append({
                "image": str(
                    image_path.relative_to(IDD_DIR)
                ),
                "detections": detections
            })

        # Progress indicator
        if (image_index + 1) % 25 == 0:

            print(
                f"Processed {image_index + 1}/"
                f"{len(benchmark_images)} images"
            )

total_end = time.perf_counter()

total_elapsed = total_end - total_start

print("\nBenchmark completed!")


# ==========================================
# 8. CALCULATE PERFORMANCE METRICS
# ==========================================

total_operations = len(latencies)

expected_operations = (
    len(benchmark_images) * MEASUREMENT_ROUNDS
)

if total_operations != expected_operations:

    raise RuntimeError(
        f"Expected {expected_operations} operations, "
        f"but completed {total_operations}."
    )

average_latency = statistics.mean(latencies)

median_latency = statistics.median(latencies)

p95_latency = float(
    np.percentile(latencies, 95)
)

throughput = (
    total_operations / total_elapsed
)

ram_after = (
    process.memory_info().rss / (1024 ** 2)
)

# Current process RAM after benchmark
ram_observed = ram_after

print("\n========== VISIONPORT BASELINE RESULTS ==========")

print(
    "Total inference operations:",
    total_operations
)

print(
    "Average latency:",
    round(average_latency, 2), "ms"
)

print(
    "Median latency:",
    round(median_latency, 2), "ms"
)

print(
    "P95 latency:",
    round(p95_latency, 2), "ms"
)

print(
    "Throughput:",
    round(throughput, 2), "images/sec"
)

print(
    "Process RAM before benchmark:",
    round(ram_before, 2), "MB"
)

print(
    "Process RAM after benchmark:",
    round(ram_observed, 2), "MB"
)


# ==========================================
# 9. GPU MEMORY
# ==========================================

if torch.cuda.is_available():

    gpu_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    gpu_reserved = (
        torch.cuda.max_memory_reserved()
        / (1024 ** 2)
    )

    print(
        "GPU peak allocated memory:",
        round(gpu_memory, 2), "MB"
    )

    print(
        "GPU peak reserved memory:",
        round(gpu_reserved, 2), "MB"
    )

else:

    gpu_memory = None
    gpu_reserved = None

    print("GPU memory: N/A - CUDA unavailable")

print("=" * 50)


# ==========================================
# 10. SAVE PREDICTIONS JSON
# ==========================================

PREDICTION_FILE = (
    RESULT_DIR / "predictions.json"
)

with open(
    PREDICTION_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        prediction_records,
        f,
        indent=4
    )

print("\nPredictions saved:", PREDICTION_FILE)

print(
    "Images with prediction records:",
    len(prediction_records)
)


# ==========================================
# 11. SAVE CSV REPORT
# ==========================================

report = {

    "Model": MODEL_PATH.name,

    "Dataset": "IDD",

    "Images": len(benchmark_images),

    "Inference Operations": total_operations,

    "Average Latency (ms)": round(
        average_latency, 2
    ),

    "Median Latency (ms)": round(
        median_latency, 2
    ),

    "P95 Latency (ms)": round(
        p95_latency, 2
    ),

    "Throughput (images/sec)": round(
        throughput, 2
    ),

    "RAM Before (MB)": round(
        ram_before, 2
    ),

    "RAM After (MB)": round(
        ram_observed, 2
    ),

    "GPU": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ),

    "GPU Memory Allocated (MB)": (
        round(gpu_memory, 2)
        if gpu_memory is not None
        else "N/A"
    ),

    "GPU Memory Reserved (MB)": (
        round(gpu_reserved, 2)
        if gpu_reserved is not None
        else "N/A"
    )
}

df = pd.DataFrame([report])

CSV_FILE = RESULT_DIR / "baseline_report.csv"

df.to_csv(CSV_FILE, index=False)

print("\nFINAL BASELINE REPORT")

print(df.to_string(index=False))

print("\nCSV report saved:", CSV_FILE)


# ==========================================
# 12. CHECK ANNOTATION FILES
# ==========================================

print("\nSearching for annotation files...")

annotation_extensions = {
    ".json", ".txt", ".yaml", ".yml", ".xml"
}

annotation_files = [
    p for p in IDD_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in annotation_extensions
]

for p in annotation_files[:100]:
    print(p.relative_to(IDD_DIR))

print(
    "\nTotal annotation/configuration files:",
    len(annotation_files)
)

print("\nVISIONPORT BENCHMARK FINISHED!")
