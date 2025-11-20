import io
from PIL import Image
import supervision as sv
from rfdetr import RFDETRSegPreview

MODEL_PATH = "model/weights.pt"
IMAGE_PATH = "D:\Adip\Object Counting\Cropped\cropps\IMG-20251010-WA0031_cage_wheel_ring_1.jpg"
CLASS_NAME = "cage_wheel_ring"
THRESHOLD_CONF = 0.60
THRESHOLD_NMS = 0.4
MIN_MASK_AREA = 1000  

model = RFDETRSegPreview(pretrain_weights=MODEL_PATH)
model.optimize_for_inference()

image = Image.open(IMAGE_PATH)

detections = model.predict(image, threshold=THRESHOLD_CONF)

detections = detections.with_nms(threshold=THRESHOLD_NMS, class_agnostic=True)

if len(detections) > 0 and hasattr(detections, "mask"):
    mask_areas = [mask.sum() for mask in detections.mask]
    keep_indices = [i for i, area in enumerate(mask_areas) if area > MIN_MASK_AREA]
    detections = detections[keep_indices]

detections.class_id = [0] * len(detections)
CUSTOM_CLASSES = [CLASS_NAME]

labels = [
    f"{CUSTOM_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.MaskAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)

object_count = len(detections)
total_mask_area = sum(mask.sum() for mask in detections.mask) if len(detections) > 0 else 0

print(f"Detected objects: {object_count}")
print(f"Total segmented mask area: {total_mask_area}")
print(f"Average mask area per object: {total_mask_area / object_count if object_count > 0 else 0:.2f}")
