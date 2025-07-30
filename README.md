# MaskDetection

## Folder Structure

```
MaskDetection/
│── data/               # 
│── models/             # 
│── notebooks/          # Jupyter experiments
│── results/            # Training logs & outputs (from Notebook/runs)
│── app.py              # Run detection app
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
│── FaceMask.v2i.yolov11/ # Main dataset for training/testing
```

## Data
- The dataset used is in `FaceMask.v2i.yolov11/` (contains `train/`, `valid/`, `test/`, and `data.yaml`).
- All other datasets have been removed.

## Training & Testing
- YOLO training and testing scripts are in separate files: `train_yolo.py` and `test_yolo.py`.
- Results from `Notebook/runs` have been moved to the `results/` directory.

## Usage
- To train: `python train_yolo.py`
- To test: `python test_yolo.py`
- To run the detection app: `python app.py`

## Requirements
Install dependencies with:
```
pip install -r requirements.txt
```

## Results
- Training logs, weights, and outputs are in the `results/` directory. 

## Modular Code Structure

```
MaskDetection/
│── app.py              # Flask app entry point
│── train_yolo.py       # Training script entry point
│── test_yolo.py        # Testing script entry point
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
│── models/             # Model loading, prediction, and utility code
│   ├── __init__.py
│   ├── yolo_utils.py   # YOLO model loading, prediction, and result parsing
│── FaceMask.v2i.yolov11/ # Main dataset
```

- All model-related logic is in `models/` for reuse and clarity.
- Scripts (`app.py`, `train_yolo.py`, `test_yolo.py`) are thin entry points.
- Utility functions and classes are separated for maintainability. 
