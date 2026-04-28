# 📄 PaperVision

Paper detection using Faster R-CNN with PyTorch.

---

## 🧠 Model

- **Architecture:** Faster R-CNN
- **Backbone options:** ResNet50-FPN (default) or MobileNetV3-Large-FPN
- **Annotation format:** YOLO format (exported from CVAT)
- **Task:** Single-class object detection (paper/document detection)

---

## 🖥️ Environment

| Item | Detail |
|------|--------|
| Platform | AI SAMK Lab (remote server) |
| GPU | NVIDIA A100 SXM4 80GB |
| CPU | AMD EPYC 7742 (128 cores) |
| RAM | 1007 GB |
| Python | 3.12.10 (local: macOS) |
| PyTorch | with CUDA support |

> ⚠️ Development is done on **macOS** (no local GPU). All training is executed on **AI SAMK Lab** via SSH.

---

## 📁 Project Structure

```
PaperVision/
├── data/
│   ├── CSVs/
│   │   ├── dataset.csv         # Full paired image-label paths
│   │   ├── train_df.csv        # 70% training split
│   │   └── val_df.csv          # 30% validation split
│   ├── images/                 # 210 images (.gitignore)
│   └── labels/                 # 210 YOLO .txt annotation files (.gitignore)
├── output_images/              # Visualized batch samples (saved during training)
│   ├── batch0_sample1.jpg
│   └── ...
├── sessions/
│   ├── best_model.pth          # Best model checkpoint (lowest val loss)
│   ├── learning_curve.png      # Train/Val loss curve plot
│   └── training_log.csv        # Per-epoch loss log
├── test_images/                # New test images for inference (not in training set)
├── test_results/               # Inference output images with bounding boxes
│   ├── result_01_...jpg
│   ├── ...
│   └── _SUMMARY_ALL_RESULTS.jpg
├── args.py                     # Hyperparameter definitions (argparse)
├── augmentations.py            # Custom augmentation pipeline (bbox-aware)
├── data_preparation.py         # Build dataset.csv + train/val split
├── dataset.py                  # PyTorch Dataset class (ObjDetectionDataset)
├── df_gen.py                   # (Reserved / utility)
├── gpu_test.py                 # Verify CUDA availability
├── inference.py                # Run inference on new images, save results with bounding boxes
├── main.py                     # Entry point: data loading + training pipeline
├── model.py                    # Faster R-CNN model builder
├── trainer.py                  # train_model() + validate_model() functions
├── utils.py                    # Box resize, batch visualization, loss curve plot
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Hyperparameters (args.py)

| Argument | Default | Options |
|----------|---------|---------|
| `--backbone` | `fasterrcnn_resnet50_fpn` | `fasterrcnn_mobilenet_v3` |
| `--num_classes` | `1` | — |
| `--image_size` | `512` | — |
| `--batch_size` | `8` | 8, 16, 32, 64 |
| `--epochs` | `25` | — |
| `--lr` | `5e-5` | — |
| `--wd` | `5e-4` | — |
| `--csv_dir` | `./data/CSVs` | — |
| `--out_dir` | `./sessions` | — |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Prepare data
```bash
python3 data_preparation.py
```
Creates `dataset.csv`, `train_df.csv`, `val_df.csv` in `data/CSVs/`.

### 3. Verify GPU (optional)
```bash
python3 gpu_test.py
```

### 4. Train
```bash
python3 main.py
```

Override defaults via command line:
```bash
python3 main.py --epochs 50 --lr 0.0001 --backbone fasterrcnn_mobilenet_v3
```

### 5. Run inference on new images
```bash
python3 inference.py \
    --model_path sessions/best_model.pth \
    --test_dir test_images/ \
    --threshold 0.75
```

Results are saved in `test_results/` with bounding boxes and confidence scores drawn on each image. A summary grid of all results is saved as `_SUMMARY_ALL_RESULTS.jpg`.

---

## 📊 Training Outputs

After training, the following are saved in `sessions/`:

| File | Description |
|------|-------------|
| `best_model.pth` | Model weights with lowest validation loss |
| `training_log.csv` | Epoch-by-epoch train/val loss |
| `learning_curve.png` | Visual plot of training progress |

Batch visualization images are saved in `output_images/` during training.

---

## 📦 Dataset

- **210 images** collected manually (200 paper images + 10 hard negative images)
- Annotated with **CVAT** (exported in YOLO format)
- Hard negative examples include scenes with computer screens and no paper present
- **Train / Val split:** 70% / 30% (random, via `sklearn.model_selection.train_test_split`)

---

## 🔧 Augmentations (augmentations.py)

Training augmentations applied on-the-fly (all bbox-aware):

- Resize to target image size
- Horizontal flip (p=0.5)
- One of: Scale, Translate, Rotate, Shear, RandomResizedCrop, RandomZoomOut (p=0.8)
- One of: ColorJitter, GaussianBlur, RandomGrayscale, RandomSharpness (p=0.6)

Validation uses only Resize + ToTensor (no augmentation).