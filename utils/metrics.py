import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_topk_accuracy(predictions, labels, k=5):
    """
    Tính toán Top-k Accuracy

    Parameters:
    - predictions: Tensor hoặc numpy array chứa các xác suất dự đoán
    - labels: Tensor hoặc numpy array chứa nhãn thực tế
    - k: Số lượng dự đoán hàng đầu để kiểm tra (mặc định là 5)

    Returns:
    - Top-k accuracy
    """
    # Chuyển đổi sang numpy nếu là tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Lấy top-k indices cho mỗi mẫu
    top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]

    # Số lượng mẫu đúng
    correct_predictions = 0

    for i in range(len(labels)):
        # Kiểm tra xem nhãn thực tế có nằm trong k dự đoán hàng đầu không
        if labels[i] in top_k_predictions[i]:
            correct_predictions += 1

    # Tính accuracy
    accuracy = correct_predictions / len(labels)

    return accuracy


def calculate_comprehensive_metrics(model, test_loader, device):
    """
    Tính toán các metric chi tiết trên toàn bộ test set

    Parameters:
    - model: Mô hình được huấn luyện
    - test_loader: DataLoader của tập test
    - device: Thiết bị tính toán (cuda/cpu)

    Returns:
    - Dictionary chứa các metric
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for joints, bones, velocities, labels in test_loader:
            # Di chuyển dữ liệu đến thiết bị
            joints = joints.to(device)
            bones = bones.to(device)
            velocities = velocities.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(joints, bones, velocities)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Lấy dự đoán
            _, predicted = torch.max(outputs, 1)

            # Lưu kết quả
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Chuyển sang numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Tính các metric
    top1_acc = calculate_topk_accuracy(all_probs, all_labels, k=1)
    top5_acc = calculate_topk_accuracy(all_probs, all_labels, k=5)

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    confusion_mat = confusion_matrix(all_labels, all_preds)

    return {
        'Top-1 Accuracy': top1_acc * 100,
        'Top-5 Accuracy': top5_acc * 100,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': confusion_mat
    }


def print_and_save_metrics(metrics, result_dir):
    """
    In và lưu các metric ra file

    Parameters:
    - metrics: Dictionary các metric
    - result_dir: Thư mục lưu kết quả
    """
    import os
    import numpy as np

    print("\n--- Comprehensive Model Metrics ---")
    for metric_name, metric_value in metrics.items():
        if metric_name != 'Confusion Matrix':
            print(f"{metric_name}: {metric_value}")

    # Lưu confusion matrix
    np.save(os.path.join(result_dir, 'confusion_matrix.npy'), metrics['Confusion Matrix'])

    # Lưu các metric khác
    with open(os.path.join(result_dir, 'model_metrics.txt'), 'w') as f:
        for metric_name, metric_value in metrics.items():
            if metric_name != 'Confusion Matrix':
                f.write(f"{metric_name}: {metric_value}\n")