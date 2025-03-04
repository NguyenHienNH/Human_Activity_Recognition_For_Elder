import os
import torch
from config.config import config
from data.preprocessing import prepare_dataset
from models.network import EnhancedMHGTN
from utils.metrics import calculate_comprehensive_metrics, print_and_save_metrics
from utils.visualization import visualize_training_history


def evaluate_model():
    # Chuẩn bị dataset
    _, test_loader = prepare_dataset()

    # Khởi tạo mô hình
    model = EnhancedMHGTN(
        num_joints=config['NUM_JOINTS'],
        num_classes=config['NUM_CLASSES']
    ).to(config['DEVICE'])

    # Load best model
    best_model_path = os.path.join(config['RESULT_DIR'], 'best_model.pth')

    if not os.path.exists(best_model_path):
        print(f"Model file not found at {best_model_path}")
        return

    best_model_checkpoint = torch.load(best_model_path, map_location=config['DEVICE'])
    model.load_state_dict(best_model_checkpoint['model_state_dict'])

    print(
        f"Loaded model from epoch {best_model_checkpoint['epoch']} with accuracy {best_model_checkpoint['accuracy'] * 100:.2f}%")

    # Tính toán và in metric
    print("\nComputing Comprehensive Metrics...")
    metrics = calculate_comprehensive_metrics(model, test_loader, config['DEVICE'])
    print_and_save_metrics(metrics, config['RESULT_DIR'])

    # Trực quan hóa lịch sử huấn luyện
    visualize_training_history(config['RESULT_DIR'])

    return metrics


if __name__ == "__main__":
    evaluate_model()