import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_training_history(result_dir):
    # Đọc log huấn luyện
    log_path = os.path.join(result_dir, 'training_log.npy')
    if not os.path.exists(log_path):
        print("Training log not found.")
        return

    training_log = np.load(log_path, allow_pickle=True).item()

    # Vẽ biểu đồ accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(training_log['train_acc'], label='Train Accuracy')
    plt.plot(training_log['test_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_log['train_loss'], label='Train Loss')
    plt.plot(training_log['test_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_history.png'))
    plt.close()

    # Vẽ biểu đồ F1, Precision, Recall
    plt.figure(figsize=(8, 6))
    plt.plot(training_log['f1'], label='F1 Score')
    plt.plot(training_log['precision'], label='Precision')
    plt.plot(training_log['recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('F1, Precision, Recall Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'metric_history.png'))
    plt.close()