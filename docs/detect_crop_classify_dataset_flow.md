# Detect, crop, classify, and build datasets

```mermaid
flowchart TD
    A[Video dau vao] --> F[Doc tung frame]
    F --> B[Detect visible person]
    B --> C[Crop tung bbox]
    C --> D1
    C --> D2
    D1[Model ngu gat]
    D2[Model raise hand]
    D1 --> E{Gop ket qua}
    D2 --> E
    E -- raisehand model = class 2 --> R[raisehand: class 2]
    E -- khong raisehand, sleep model = class 1 --> S[sleep: class 1]
    E -- con lai --> N[normal: class 0]
    R --> Y[dataset_yolo: anh goc + bbox]
    S --> Y
    N --> Y
    R --> K[dataset_cls: crop theo class]
    S --> K
    N --> K
```

## Output

```text
output/
  dataset_yolo/
    images/
    labels/
    data.yaml
  dataset_cls/
    normal/
    sleep/
    raisehand/
  visualize/                 # khi dung --visualize
  metadata.csv
```

`dataset_yolo/labels` dung bbox `visible person` tren anh goc voi class `0=normal`,
`1=sleep`, `2=raisehand`. Neu hai model cung duong tinh, `raisehand` duoc uu tien.

Frame video duoc dat ten theo mau `<ten_video>_<frame_index>.jpg`, vi du
`camera01_00000015.jpg`. Dung `--visualize` de luu frame co bbox va nhan du doan.
