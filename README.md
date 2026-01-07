# Fire Detection Using Deep Learning

## Overview
This project implements a **fire detection system** using **deep learning and transfer learning**.
A pretrained **InceptionV3 CNN** model is fine-tuned on a custom fire dataset and used to classify
images and GIF videos into:
- Fire
- No Fire
- Start Fire

The system also performs **frame-wise fire detection on GIF/video inputs** and generates
annotated outputs.


##  Model Used
- InceptionV3 (pretrained on ImageNet)
- Fine-tuned using a custom dataset


##  Dataset Structure
```

dataset/
├── fire/
├── no_fire/
└── start_fire/

````

## Video/GIF Fire Detection
- Input: GIF or video files
- Output: Annotated GIF with predicted class and confidence


##  Technologies Used
- Python 3.13
- TensorFlow / Keras
- OpenCV
- ImageIO
- NumPy


## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Train the model

```bash
python transfer_learning.py
```

### 3. Run GIF fire detection

```bash
python video_annotation.py
```


##Results

The model successfully detects fire-related patterns and annotates GIF frames with
classification labels and confidence scores. Due to limited dataset size, probabilities
reflect reasonable uncertainty, which is expected behavior.


## Notes

* This project is intended for **educational and internship purposes**
* Accuracy can be improved with larger datasets


## Author

**Supriya Kulkarni**
(**Intern ID:SMI82128**)
