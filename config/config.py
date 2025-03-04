import os
import torch

config = {
    'DATA_DIR': '/content/drive/MyDrive/Colab Notebooks/ETRI-Activity3D_OpenPose',
    'RESULT_DIR': '/content/drive/MyDrive/Colab Notebooks/resultFinal15',
    'NUM_JOINTS': 25,
    'NUM_CLASSES': 55,
    'NUM_FRAMES': 100,
    'BATCH_SIZE': 64,
    'MAX_EPOCHS': 300,
    'INIT_LR': 0.0005,
    'HIDDEN_DIM': 128,
    'DROPOUT': 0.6,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}