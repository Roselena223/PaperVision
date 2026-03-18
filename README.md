# 📄 PaperVision

Paper detection using Faster R-CNN with PyTorch.

## Model
- Faster R-CNN + ResNet50 / MobileNetV3 backbone
- YOLO format annotations (CVAT export)

## Project Structure
PaperVision/
├── data/
│   ├── CSVs/
│   │   ├── dataset.csv
│   │   ├── train_df.csv
│   │   └── val_df.csv
│   ├── images/          # initial images (add into .gitignore)
│   └── labels/          # file .txt YOLO annotation
├── sessions/
│   ├── args.py
│   ├── data_preparation.py
│   ├── dataset.py
│   ├── df_gen.py
│   ├── gpu_test.py
│   ├── main.py
│   ├── model.py
│   └── trainer.py
├── .gitignore
├── README.md
└── requirements.txt

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare data: `python sessions/data_preparation.py`
3. Train: `python sessions/main.py`

## Dataset
- 200 images collected manually
- Annotated with CVAT (YOLO format)