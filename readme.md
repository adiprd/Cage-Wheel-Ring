# RF-DETR Object Detection & Segmentation Pipeline

Dokumentasi Proyek

README ini disusun untuk memberikan penjelasan menyeluruh mengenai implementasi sistem deteksi dan segmentasi objek berbasis **RF-DETR (Region-Free DEtection TRansformer)**. Proyek ini mencakup proses pemuatan model, inferensi pada data gambar lokal, pengolahan hasil prediksi, hingga visualisasi menggunakan library **Supervision**.

Tujuan dari proyek ini adalah menyediakan workflow inferensi yang stabil dan dapat direplikasi, terutama untuk kebutuhan seperti penghitungan objek (object counting), inspeksi visual, analisis kualitas produk, atau otomasi berbasis visi komputer.

---

## 1. Latar Belakang Proyek

Dalam berbagai kebutuhan industri, proses deteksi objek tidak hanya membutuhkan bounding box, tetapi juga segmentasi mask untuk akurasi yang lebih tinggi. Model RF-DETR dipilih karena:

1. Arsitektur transformer yang efisien dalam inference.
2. Kemampuan untuk memberikan **segmentation mask** dan **bounding box** bersamaan.
3. Lebih robust terhadap skala objek dan kondisi pencahayaan.
4. Dapat dipercepat dengan fungsi `optimize_for_inference()`.

Proyek ini dirancang untuk:

* Mengolah gambar dari folder lokal.
* Menjalankan inference dengan threshold kepercayaan tertentu.
* Menampilkan hasil anotasi secara visual.
* Menghasilkan label otomatis berdasarkan dataset COCO.

---

## 2. Struktur Proyek

Struktur dasar proyek adalah sebagai berikut:

```
project/
│
├── model/
│   └── weights.pt                 # Bobot pretrained RF-DETR
│
├── scripts/
│   └── infer.py                   # Script utama inferensi
│
├── data/
│   └── samples/                   # Gambar-gambar uji
│
├── outputs/
│   └── annotated/                 # Output gambar setelah anotasi
│
└── README.md
```

Pengguna dapat menyesuaikan struktur sesuai kebutuhan selama path menuju bobot model dan input image tetap benar.

---

## 3. Instalasi Dependencies

Semua library yang diperlukan dapat dipasang melalui:

```bash
pip install rfdetr supervision pillow requests
```

Pastikan Python minimal versi 3.8.

---

## 4. Pipeline Kerja Sistem

Berikut penjelasan pipeline inferensi yang digunakan dalam proyek ini:

1. **Load Model RF-DETR**
   Menggunakan `RFDETRSegPreview` dengan bobot pretrained dari direktori lokal.

2. **Optimasi Model untuk Inference**
   `model.optimize_for_inference()` bertujuan mengurangi overhead dan mempercepat prediksi.

3. **Load Gambar Input**
   Gambar di-load menggunakan PIL dari path lokal. Tidak ada preprocessing tambahan, karena RF-DETR menangani normalisasi internal.

4. **Inferensi Deteksi dan Segmentasi**
   Pemanggilan:

   ```python
   detections = model.predict(image, threshold=0.4)
   ```

   Menghasilkan:

   * class ID
   * bounding box
   * segmentation mask
   * confidence score

5. **Generate Label**
   Label dibentuk menggunakan mapping COCO:

   Format:
   `"class_name confidence"`

6. **Anotasi Visual**
   Library Supervision digunakan untuk menggambar:

   * segmentation mask
   * label box
   * class name & confidence

7. **Display Output**
   Hasil anotasi divisualisasikan dengan `sv.plot_image()`.

---

## 5. Kode Lengkap (Script Inferensi)

```python
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSegPreview
from rfdetr.util.coco_classes import COCO_CLASSES

# Load model
model = RFDETRSegPreview(pretrain_weights='model/weights.pt')
model.optimize_for_inference()

# Load image
url = r"D:\Adip\Object Counting\Cropped\crops\cage_wheel_ring_0.jpg"
image = Image.open(url)

# Predict
detections = model.predict(image, threshold=0.4)

# Generate labels
labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

# Annotate output
annotated_image = image.copy()
annotated_image = sv.MaskAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# Display
sv.plot_image(annotated_image)
```

---

## 6. Penjelasan Teknis Tambahan

### a. Threshold Confidence

Nilai threshold 0.4 dipilih agar model menampilkan objek dengan tingkat kepercayaan menengah. Threshold dapat dinaikkan untuk mengurangi false positive atau diturunkan untuk mendeteksi objek kecil.

### b. Segmentasi Mask

Mask dihasilkan dalam format polygon atau binary map, kemudian ditumpuk di atas gambar asli oleh Supervision.

### c. Kompatibilitas COCO Classes

RF-DETR bawaan menggunakan mapping 80 kelas COCO. Jika model Anda adalah model custom training, mapping dapat diganti sesuai label dataset pelatihan.

### d. Kinerja Inference

`optimize_for_inference()` menyederhanakan pipeline internal:

* mempercepat eksekusi
* menghapus komponen yang hanya dibutuhkan saat training
* mengurangi penggunaan memori

---

## 7. Pengembangan Lanjutan

Proyek ini dapat diperluas ke beberapa arah:

1. **Object Counting Otomatis**
   Menghitung jumlah objek per kelas dari `detections.class_id`.

2. **Export Hasil ke File**
   Menyimpan output ke direktori `/outputs/annotated`.

3. **Batch Processing**
   Memproses satu folder secara otomatis.

4. **Integrasi Kamera Real-Time**
   Menjalankan inference frame-by-frame dari webcam atau CCTV.

5. **Training Ulang RF-DETR**
   Untuk dataset custom di lingkungan industri.

---

## 8. Lisensi

Penggunaan RF-DETR dan library terkait mengikuti lisensi resmi masing-masing paket. Proyek ini dapat dikembangkan bebas sepanjang mengikuti aturan library upstream.

---

Jika Anda ingin, saya bisa menambahkan:

* Flowchart arsitektur pipeline
* Contoh hasil visual inferensi
* Versi README dalam bahasa Inggris
* Penjelasan cara melakukan object tracking dengan Supervision

Cukup beri instruksi.
