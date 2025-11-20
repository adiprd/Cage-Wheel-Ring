Object Detection & Segmentation System Documentation
Overview
Sistem ini menggunakan model RFDETR (Recursive Feature Pyramid Detection Transformer) untuk melakukan object detection dan instance segmentation pada gambar. Model ini dapat mendeteksi objek dengan bounding box dan mask segmentation dengan akurasi tinggi.

Technical Specifications
Model Architecture
Model: RFDETRSegPreview (Recursive Feature Pyramid DETR dengan Segmentation)

Type: Transformer-based object detection dengan instance segmentation

Input: RGB Images

Output: Bounding boxes, confidence scores, dan segmentation masks

System Requirements
python
# Dependencies yang diperlukan
Python 3.8+
PyTorch 1.12+
torchvision 0.13+
supervision 0.10+
Pillow (PIL)
Code Implementation
Import Libraries
python
import io
from PIL import Image
import supervision as sv
from rfdetr import RFDETRSegPreview
Configuration Parameters
python
# Path konfigurasi
MODEL_PATH = "model/weights.pt"                    # Path ke model weights
IMAGE_PATH = "D:\\Adip\\Object Counting\\Cropped\\cropps\\IMG-20251010-WA0031_cage_wheel_ring_1.jpg"  # Path gambar input
CLASS_NAME = "cage_wheel_ring"                     # Nama kelas objek yang dideteksi

# Threshold konfigurasi
THRESHOLD_CONF = 0.60      # Confidence threshold untuk deteksi
THRESHOLD_NMS = 0.4        # Threshold untuk Non-Maximum Suppression
MIN_MASK_AREA = 1000       # Area minimum mask (dalam pixels)
Model Initialization
python
# Inisialisasi model
model = RFDETRSegPreview(pretrain_weights=MODEL_PATH)

# Optimasi model untuk inference
model.optimize_for_inference()
Image Processing Pipeline
python
# Load gambar
image = Image.open(IMAGE_PATH)

# Deteksi objek
detections = model.predict(image, threshold=THRESHOLD_CONF)

# Apply Non-Maximum Suppression
detections = detections.with_nms(threshold=THRESHOLD_NMS, class_agnostic=True)

# Filter berdasarkan area mask minimum
if len(detections) > 0 and hasattr(detections, "mask"):
    mask_areas = [mask.sum() for mask in detections.mask]
    keep_indices = [i for i, area in enumerate(mask_areas) if area > MIN_MASK_AREA]
    detections = detections[keep_indices]

# Set class ID dan label
detections.class_id = [0] * len(detections)
CUSTOM_CLASSES = [CLASS_NAME]

# Generate labels untuk annotation
labels = [
    f"{CUSTOM_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]
Visualization and Annotation
python
# Buat annotated image
annotated_image = image.copy()

# Annotasi mask segmentation
annotated_image = sv.MaskAnnotator().annotate(annotated_image, detections)

# Annotasi label dengan confidence score
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# Tampilkan hasil
sv.plot_image(annotated_image)
Results Analysis
python
# Analisis hasil deteksi
object_count = len(detections)
total_mask_area = sum(mask.sum() for mask in detections.mask) if len(detections) > 0 else 0

# Print hasil
print(f"Detected objects: {object_count}")
print(f"Total segmented mask area: {total_mask_area}")
print(f"Average mask area per object: {total_mask_area / object_count if object_count > 0 else 0:.2f}")
Parameter Tuning Guide
Confidence Threshold (THRESHOLD_CONF)
Range: 0.0 - 1.0

Rekomendasi: 0.50 - 0.75

Efek:

Nilai tinggi: Lebih sedikit deteksi, precision lebih tinggi

Nilai rendah: Lebih banyak deteksi, recall lebih tinggi

NMS Threshold (THRESHOLD_NMS)
Range: 0.0 - 1.0

Rekomendasi: 0.3 - 0.5

Efek: Mengurangi overlapping detections

Minimum Mask Area (MIN_MASK_AREA)
Unit: Pixels

Rekomendasi: 500 - 2000 (tergantung resolusi gambar)

Efek: Filter deteksi berdasarkan ukuran objek

Output Format
Detections Object
Struktur output dari model.predict():

python
detections = {
    'xyxy': array([[x1, y1, x2, y2], ...]),  # Bounding boxes
    'confidence': array([0.95, 0.87, ...]),   # Confidence scores
    'class_id': array([0, 0, ...]),           # Class IDs
    'mask': array([mask1, mask2, ...])        # Segmentation masks
}
Metrics Output
Object Count: Jumlah objek yang terdeteksi

Total Mask Area: Total area semua mask (pixels)

Average Mask Area: Rata-rata area per objek (pixels)

Performance Optimization
Model Optimization
python
# Optimasi untuk inference
model.optimize_for_inference()

# Untuk GPU acceleration (jika tersedia)
model.to('cuda')  # Pindah model ke GPU
Memory Management
python
# Clear cache setelah inference
import torch
torch.cuda.empty_cache()  # Jika menggunakan GPU

# Batch processing untuk multiple images
def process_batch(image_paths):
    results = []
    for image_path in image_paths:
        image = Image.open(image_path)
        detections = model.predict(image, threshold=THRESHOLD_CONF)
        results.append(detections)
    return results
Error Handling
Common Issues and Solutions
python
try:
    # Load model
    model = RFDETRSegPreview(pretrain_weights=MODEL_PATH)
except FileNotFoundError:
    print(f"Model weights not found at {MODEL_PATH}")
    exit(1)

try:
    # Load image
    image = Image.open(IMAGE_PATH)
except FileNotFoundError:
    print(f"Image not found at {IMAGE_PATH}")
    exit(1)

# Check jika detections ada
if len(detections) == 0:
    print("No objects detected. Adjust threshold parameters.")
Usage Examples
Basic Usage
python
# Single image processing
from object_detection_system import ObjectDetectionSystem

detector = ObjectDetectionSystem(
    model_path="model/weights.pt",
    class_name="cage_wheel_ring",
    confidence_threshold=0.60
)

results = detector.detect("path/to/image.jpg")
detector.visualize_results()
Batch Processing
python
# Multiple images
image_paths = [
    "image1.jpg",
    "image2.jpg", 
    "image3.jpg"
]

for image_path in image_paths:
    detections = model.predict(Image.open(image_path), threshold=THRESHOLD_CONF)
    print(f"Detected {len(detections)} objects in {image_path}")
Deployment Considerations
Production Requirements
Memory: Minimum 4GB RAM (8GB recommended)

Storage: 2GB untuk model dan dependencies

GPU: Optional, tetapi mempercepat inference 5-10x

Scalability
python
# Multi-threading untuk high-throughput
from concurrent.futures import ThreadPoolExecutor

def process_image(image_path):
    image = Image.open(image_path)
    return model.predict(image, threshold=THRESHOLD_CONF)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, image_paths))
Maintenance and Updates
Model Updates
Periodically retrain model dengan data baru

Monitor performance metrics

Update threshold parameters berdasarkan kebutuhan

Performance Monitoring
python
# Logging untuk monitoring
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log detection results
logger.info(f"Detection completed: {len(detections)} objects found")
logger.info(f"Average confidence: {detections.confidence.mean():.3f}")
Dokumentasi ini memberikan panduan lengkap untuk menggunakan sistem object detection dan segmentation berdasarkan kode yang diberikan. Sistem ini cocok untuk aplikasi industrial inspection, quality control, dan automated object counting.

