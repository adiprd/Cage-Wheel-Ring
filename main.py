import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSegPreview
from rfdetr.util.coco_classes import COCO_CLASSES

# Load model
model = RFDETRSegPreview(pretrain_weights='model/weights.pt')
model.optimize_for_inference()

# Load image
url = "D:\Adip\Object Counting\Cropped\crops\cage_wheel_ring_0.jpg"
image = Image.open(url)

# Predict dengan threshold
detections = model.predict(image, threshold=0.4)  

# Labels
labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

# Annotate
annotated_image = image.copy()
annotated_image = sv.MaskAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)
