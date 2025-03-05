# Skeleton-Based Action Recognition
This repository contains a deep learning framework for skeleton-based human action recognition using a multi-stream graph transformer network.

## Overview
The framework utilizes a multi-stream approach with joints, bones, and motion features to capture the spatial-temporal dynamics of human actions. The model incorporates several advanced components:
* Multi-scale feature extractors
* Shift graph convolution for skeleton modeling
* Swin Transformer layers for temporal modeling
* Coordinate Attention for feature refinement
* Cross-stream attention for inter-stream interactions
* SENet fusion for effective multi-stream fusion

## Model Architecture
The Enhanced Multi-stream Graph Transformer Network (EnhancedMHGTN) processes three input streams:
1. **Joint positions** (spatial configuration)
2. **Bone features** (relative positions)
3. **Velocity features** (temporal dynamics)

Each stream goes through feature extraction, graph modeling, and transformer-based temporal modeling before being fused together.

## Requirements
```bash
torch>=1.10.0
numpy>=1.19.5
scipy>=1.7.0
scikit-learn>=0.24.2
matplotlib>=3.4.3
tqdm>=4.62.3
```

## Installation
```bash
git clone https://github.com/yourusername/skeleton-action-recognition.git
cd skeleton-action-recognition
pip install -r requirements.txt
```

## Dataset Preparation
The code expects skeleton data in .mat format with the following structure:
* Each file contains skeleton joint coordinates.
* Files are organized by person ID for cross-subject validation.
* File naming follows a pattern with action class encoded in the name.

### **ETRI Activity3D Dataset**
This framework is designed to support the **ETRI-Activity3D dataset**, which is a large-scale dataset for skeleton-based action recognition. The dataset consists of:
* **55,000 action samples** collected from **112 subjects**.
* **55 action categories**, including daily activities, gestures, and interactions.
* **3D skeleton data** captured from a **Kinect v2** sensor.
* **Cross-subject and cross-view evaluation protocols** for robust performance testing.

#### **Preprocessing ETRI Dataset**
The dataset files should be preprocessed to extract **joint coordinates**, **bone relations**, and **motion features** before feeding them into the model. Ensure that:
* Data is converted into `.mat` format.
* Labels follow the standard encoding scheme for training and evaluation.

## Usage
### Training
```bash
python main.py --mode train --data_dir /path/to/dataset --result_dir /path/to/save/results
```

### Evaluation
```bash
python main.py --mode eval --data_dir /path/to/dataset --result_dir /path/to/results
```

### Both Training and Evaluation
```bash
python main.py --mode train_eval --data_dir /path/to/dataset --result_dir /path/to/save/results
```

### Additional Parameters
* `--batch_size`: Training batch size (default: 64)
* `--epochs`: Maximum training epochs (default: 300)
* `--lr`: Initial learning rate (default: 0.0005)
* `--seed`: Random seed (default: 42)

## Results
The model achieves the following performance metrics:
* **Training Accuracy**: **92.45%**
* **Testing Accuracy**: **85.19%**

Evaluation metrics include:
* **Top-1 Accuracy**: **85.19%**
* **Top-5 Accuracy**: **98.63%**
* **Precision**: **85.35%**
* **Recall**: **84.15%**
* **F1-score**: **84.56%**

Training curves are automatically generated and saved to the results directory.

## Citation
If you use this code for your research, please cite our work:
```bibtex
@article{your_reference_here,
 
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

