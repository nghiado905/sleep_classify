# Human Detection (2 Classes: Head & Visible-Person)

Gói module gọn nhẹ (standalone & portable) cho bài toán **Human Detection**, chỉ bao gồm các thành phần cốt lõi để huấn luyện và dự đoán, dễ dàng copy/di chuyển sang môi trường hoặc máy chủ khác.

---

## 1. Cấu Trúc Thư Mục

```text
human-detection/
├── README.md                          # Tài liệu hướng dẫn chi tiết
├── requirements.txt                   # Danh sách thư viện phụ thuộc
├── data.yaml.template                 # Template cấu hình dữ liệu YOLO
├── models/
│   └── human_detection_2class.pt      # Model YOLO11n đã train 2 class (~5.3MB)
├── test-img.jpg                       # Ảnh mẫu để kiểm tra nhanh
├── train.py                           # Script huấn luyện / fine-tune model
└── predict.py                         # Script dự đoán (ảnh đơn, thư mục ảnh, video)
```

---

## 2. Thông Tin Model & Classes

* **Kiến trúc:** YOLO11n
* **Kích thước file trọng số:** ~5.3 MB
* **Số lượng Classes:** **2 classes**
  * `0`: **`head`** (Phần đầu người)
  * `1`: **`visible-person`** (Phần cơ thể người nhìn thấy được)

---

## 3. Cài Đặt Môi Trường

Khuyến nghị sử dụng Python >= 3.10:

```bash
pip install -r requirements.txt
```

Hoặc cài trực tiếp:
```bash
pip install ultralytics torch torchvision opencv-python tqdm
```

---

## 4. Hướng Dẫn Chạy Predict (Suy Luận)

Script `predict.py` tự động vẽ bounding box, hiển thị tên class (`head`, `visible-person`) cùng confidence score, và lưu ảnh ra thư mục chỉ định.

### A. Chạy thử nhanh với ảnh mẫu
```bash
python predict.py
```
*(Kết quả vẽ bounding box sẽ được lưu vào thư mục `predict_results/`)*

### B. Chạy trên một ảnh bất kỳ
```bash
python predict.py --source /path/to/image.jpg --output-dir predict_results
```

### C. Chạy trên cả thư mục nhiều ảnh (Batch Prediction)
```bash
python predict.py --source /path/to/image_folder/ --output-dir predict_results --conf 0.25
```

### D. Chạy trên video
```bash
python predict.py --source /path/to/video.mp4 --output-dir predict_results
```

### Các tham số tùy chọn chính:
* `--model`: Đường dẫn model `.pt` (mặc định: `models/human_detection_2class.pt`)
* `--source`: Đường dẫn ảnh, folder ảnh hoặc video (mặc định: `test-img.jpg`)
* `--output-dir`: Thư mục lưu kết quả (mặc định: `predict_results`)
* `--conf`: Ngưỡng confidence (mặc định: `0.25`)
* `--iou`: Ngưỡng IoU Non-Maximum Suppression (mặc định: `0.7`)
* `--imgsz`: Kích thước ảnh resize khi inference (mặc định: `640`)
* `--device`: Thiết bị chạy (`"0"` cho GPU hoặc `"cpu"`)

---

## 5. Hướng Dẫn Training / Fine-tune

### Bước 1: Chuẩn bị file cấu hình dữ liệu `data.yaml`
Sao chép từ file template:
```bash
cp data.yaml.template data.yaml
```

Mở `data.yaml` và chỉnh sửa đường dẫn dataset:
```yaml
path: /duong/dan/toi/dataset
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: head
  1: visible-person
```

### Bước 2: Chạy train
* **Train mới từ pretrain YOLO11n:**
  ```bash
  python train.py --data data.yaml --epochs 100 --batch 16 --imgsz 640
  ```

* **Fine-tune tiếp từ model 2-class sẵn có:**
  ```bash
  python train.py --data data.yaml --model models/human_detection_2class.pt --epochs 50 --batch 16
  ```

* Trọng số sau khi train sẽ được tự động lưu tại: `runs/<run_name>/weights/best.pt`.
