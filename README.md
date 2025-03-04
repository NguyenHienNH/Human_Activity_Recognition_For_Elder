# Skeleton-Based Action Recognition

This repository contains a deep learning framework for skeleton-based human action recognition using a multi-stream graph transformer network.

## Overview

The framework utilizes a multi-stream approach with joints, bones, and motion features to capture the spatial-temporal dynamics of human actions. The model incorporates several advanced components:

- Multi-scale feature extractors
- Shift graph convolution for skeleton modeling
- Swin Transformer layers for temporal modeling
- Coordinate Attention for feature refinement
- Cross-stream attention for inter-stream interactions
- SENet fusion for effective multi-stream fusion

## Model Architecture

The Enhanced Multi-stream Graph Transformer Network (EnhancedMHGTN) processes three input streams:
1. Joint positions (spatial configuration)
2. Bone features (relative positions)
3. Velocity features (temporal dynamics)

Each stream goes through feature extraction, graph modeling, and transformer-based temporal modeling before being fused together.

## Requirements

```
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
- Each file contains skeleton joints coordinates 
- Files are organized by person ID for cross-subject validation
- File naming follows a pattern with action class encoded in the name

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

- `--batch_size`: Training batch size (default: 64)
- `--epochs`: Maximum training epochs (default: 300)
- `--lr`: Initial learning rate (default: 0.0005)
- `--seed`: Random seed (default: 42)

## Results

The model provides the following evaluation metrics:
- Accuracy test: 84.04%
- Accuracy test: 92.54%
- Top-1 Accuracy
- Top-5 Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix

Training curves are automatically generated and saved to the results directory.

## Citation



## License

This project is licensed under the MIT License - see the LICENSE file for details.
